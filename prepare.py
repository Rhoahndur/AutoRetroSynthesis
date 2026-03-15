"""
One-time data preparation for retrosynthesis autoresearch experiments.
Downloads USPTO-50K, builds SMILES tokenizer, extracts building blocks.

Usage:
    uv run prepare.py              # full prep
    uv run prepare.py --tiny       # tiny subset for local testing (500 reactions)

Data stored in ~/.cache/autoresearch-retro/
"""

import os
import re
import sys
import json
import time
import pickle
import argparse
from contextlib import nullcontext
from collections import Counter

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 256         # context length (reactions are short)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_SAMPLES = 500        # number of val samples for accuracy evaluation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-retro")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# SMILES tokenization regex (Schwaller et al.)
# Splits SMILES into chemically meaningful tokens:
#   multi-char atoms (Br, Cl, [nH]), single-char atoms (C, N, O),
#   bonds (=, #, -), ring digits, branches, stereo markers
SMILES_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$"
    r"|%[0-9]{2}|[0-9])"
)

# Special token IDs (fixed)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>"]

# ---------------------------------------------------------------------------
# SMILES Canonicalization (RDKit)
# ---------------------------------------------------------------------------

def _get_chem():
    """Lazy import RDKit Chem module."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    return Chem

def canonicalize_smiles(smiles):
    """Canonicalize a single SMILES string. Returns None if invalid."""
    Chem = _get_chem()
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None

def canonicalize_reaction_smiles(smiles):
    """Canonicalize multi-fragment SMILES (separated by '.').
    Fragments are individually canonicalized and sorted alphabetically."""
    if not smiles or not smiles.strip():
        return None
    fragments = smiles.split(".")
    canonical = []
    for frag in fragments:
        c = canonicalize_smiles(frag)
        if c is None:
            return None
        canonical.append(c)
    if not canonical:
        return None
    return ".".join(sorted(canonical))

# ---------------------------------------------------------------------------
# SMILES Tokenizer
# ---------------------------------------------------------------------------

class SMILESTokenizer:
    """Character-level SMILES tokenizer with regex splitting."""

    def __init__(self, token2id, id2token):
        self.token2id = token2id
        self.id2token = id2token
        self.vocab_size = len(token2id)

    @classmethod
    def from_file(cls, path=None):
        if path is None:
            path = os.path.join(DATA_DIR, "vocab.json")
        with open(path) as f:
            data = json.load(f)
        token2id = data["token2id"]
        id2token = {int(k): v for k, v in data["id2token"].items()}
        return cls(token2id, id2token)

    def save(self, path=None):
        if path is None:
            path = os.path.join(DATA_DIR, "vocab.json")
        with open(path, "w") as f:
            json.dump({
                "token2id": self.token2id,
                "id2token": {str(k): v for k, v in self.id2token.items()},
            }, f, indent=2)

    @classmethod
    def build_from_reactions(cls, reactions):
        """Build vocabulary from list of (product_smiles, reactant_smiles) pairs."""
        token_set = set()
        for product, reactants in reactions:
            token_set.update(SMILES_PATTERN.findall(product))
            token_set.update(SMILES_PATTERN.findall(reactants))

        token2id = {}
        for i, name in enumerate(SPECIAL_TOKENS):
            token2id[name] = i
        for token in sorted(token_set):
            if token not in token2id:
                token2id[token] = len(token2id)

        id2token = {v: k for k, v in token2id.items()}
        return cls(token2id, id2token)

    def encode(self, smiles):
        """Tokenize SMILES string → list of int IDs."""
        return [self.token2id[t] for t in SMILES_PATTERN.findall(smiles) if t in self.token2id]

    def decode(self, ids):
        """Decode list of int IDs → SMILES string."""
        return "".join(self.id2token[i] for i in ids if i in self.id2token and i >= len(SPECIAL_TOKENS))

    def encode_reaction(self, product_smiles, reactant_smiles):
        """Encode a reaction: <bos> [product] <sep> [reactants] <eos>"""
        return [BOS_ID] + self.encode(product_smiles) + [SEP_ID] + self.encode(reactant_smiles) + [EOS_ID]

    def get_vocab_size(self):
        return self.vocab_size

# ---------------------------------------------------------------------------
# Data download & processing
# ---------------------------------------------------------------------------

def _parse_rxn_smiles(rxn_str):
    """Parse reaction SMILES in various formats. Returns (reactants, product) or None."""
    if ">>" in rxn_str:
        parts = rxn_str.split(">>")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    elif ">" in rxn_str:
        parts = rxn_str.split(">")
        if len(parts) >= 2:
            return parts[0].strip(), parts[-1].strip()
    return None

def _find_rxn_column(row):
    """Find reaction SMILES in a dataset row (handles different column names)."""
    for key in ("rxn_smiles", "canonical_rxn", "reaction_smiles", "REACTION"):
        if key in row and isinstance(row[key], str) and ">" in row[key]:
            return row[key]
    # Fallback: try any string column with '>>'
    for val in row.values():
        if isinstance(val, str) and ">>" in val:
            return val
    return None

def download_and_process(tiny=False):
    """Download USPTO-50K, canonicalize, tokenize, and save."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if already done
    if all(os.path.exists(os.path.join(DATA_DIR, f"{s}_data.pt")) for s in ("train", "val", "test")):
        print("Data: already processed")
        tok = SMILESTokenizer.from_file()
        print(f"Tokenizer: vocab_size={tok.vocab_size}")
        return

    print("Data: downloading USPTO-50K from HuggingFace...")
    from datasets import load_dataset

    ds = None
    for name in ("pingzhili/uspto-50k", "sagawa/USPTO-50k"):
        try:
            ds = load_dataset(name)
            print(f"Data: loaded dataset '{name}'")
            break
        except Exception as e:
            print(f"  tried {name}: {e}")

    if ds is None:
        print("ERROR: could not download USPTO-50K.")
        print("Please download manually and place CSV files in", DATA_DIR)
        sys.exit(1)

    # Map split names (different datasets use different names)
    split_map = {}
    available = set(ds.keys())
    for target, candidates in [
        ("train", ["train"]),
        ("val", ["validation", "valid", "val", "dev"]),
        ("test", ["test"]),
    ]:
        for c in candidates:
            if c in available:
                split_map[target] = c
                break

    if "train" not in split_map:
        print(f"ERROR: no training split found. Available: {available}")
        sys.exit(1)

    # Parse and canonicalize reactions
    _get_chem()  # warm up RDKit
    all_reactions = {}

    for split_name, hf_split in split_map.items():
        reactions = []
        skipped = 0
        for row in ds[hf_split]:
            rxn_str = _find_rxn_column(row)
            if rxn_str is None:
                skipped += 1
                continue
            parsed = _parse_rxn_smiles(rxn_str)
            if parsed is None:
                skipped += 1
                continue
            reactants_raw, product_raw = parsed
            product = canonicalize_smiles(product_raw)
            reactants = canonicalize_reaction_smiles(reactants_raw)
            if product and reactants:
                reactions.append((product, reactants))
            else:
                skipped += 1

        if tiny:
            reactions = reactions[:500]

        all_reactions[split_name] = reactions
        print(f"Data: {split_name}: {len(reactions)} reactions ({skipped} skipped)")

    # Build tokenizer from training data
    print("Tokenizer: building vocabulary...")
    tokenizer = SMILESTokenizer.build_from_reactions(all_reactions["train"])
    tokenizer.save()
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Tokenize and save each split
    for split_name, reactions in all_reactions.items():
        sequences = []
        skipped = 0
        for product, reactants in reactions:
            seq = tokenizer.encode_reaction(product, reactants)
            if len(seq) > MAX_SEQ_LEN:
                skipped += 1
                continue
            padded = seq + [PAD_ID] * (MAX_SEQ_LEN - len(seq))
            sequences.append(padded)

        if skipped > 0:
            print(f"Data: {split_name}: {skipped} reactions exceeded MAX_SEQ_LEN={MAX_SEQ_LEN}")

        seq_tensor = torch.tensor(sequences, dtype=torch.long)

        # Compute loss masks: 1 for reactant + eos positions in targets, 0 elsewhere
        targets = seq_tensor[:, 1:]  # (N, MAX_SEQ_LEN-1)
        # Find SEP position in targets for each sequence
        has_sep = (targets == SEP_ID).any(dim=1)
        sep_positions = (targets == SEP_ID).int().argmax(dim=1)  # (N,)
        positions = torch.arange(MAX_SEQ_LEN - 1).unsqueeze(0)  # (1, T)
        after_sep = positions > sep_positions.unsqueeze(1)  # (N, T)
        not_pad = targets != PAD_ID
        loss_masks = (after_sep & not_pad & has_sep.unsqueeze(1)).float()

        data = {"sequences": seq_tensor, "loss_masks": loss_masks}
        path = os.path.join(DATA_DIR, f"{split_name}_data.pt")
        torch.save(data, path)
        print(f"Data: saved {split_name} ({len(sequences)} reactions) → {path}")

# ---------------------------------------------------------------------------
# Building blocks extraction
# ---------------------------------------------------------------------------

def extract_building_blocks():
    """Extract commonly appearing reactant fragments as building blocks."""
    bb_path = os.path.join(DATA_DIR, "building_blocks.pkl")
    if os.path.exists(bb_path):
        with open(bb_path, "rb") as f:
            bb = pickle.load(f)
        print(f"Building blocks: {len(bb)} molecules (already extracted)")
        return

    print("Building blocks: extracting from training reactions...")
    _get_chem()

    # Load training data to get raw canonical reactions
    # Re-parse from the dataset (or from saved tokenized data)
    from datasets import load_dataset

    ds = None
    for name in ("pingzhili/uspto-50k", "sagawa/USPTO-50k"):
        try:
            ds = load_dataset(name)
            break
        except Exception:
            pass

    reactant_counts = Counter()
    if ds is not None:
        train_key = "train"
        for row in ds[train_key]:
            rxn_str = _find_rxn_column(row)
            if rxn_str is None:
                continue
            parsed = _parse_rxn_smiles(rxn_str)
            if parsed is None:
                continue
            reactants_raw = parsed[0]
            for frag in reactants_raw.split("."):
                c = canonicalize_smiles(frag.strip())
                if c:
                    reactant_counts[c] += 1

    # Building blocks: reactants appearing in >= 3 different reactions
    building_blocks = {s for s, cnt in reactant_counts.items() if cnt >= 3}

    # Supplement with common lab reagents
    common = [
        "O", "CO", "CCO", "Cl", "Br", "I", "N", "[H][H]",
        "CC(=O)O", "O=CO", "CC#N", "ClCCl", "CS(C)=O",
        "c1ccccc1", "CC(=O)OC(C)=O", "CC(=O)Cl", "O=C(O)O",
        "O=S(=O)(O)O", "[Na+].[OH-]", "[Li]CCCC", "CC(C)=O",
        "C(=O)Cl", "CCN(CC)CC", "C1CCOC1", "ClC(Cl)Cl",
        "O=[N+]([O-])O", "[Na]", "[K]", "OO", "[Pd]",
    ]
    for r in common:
        c = canonicalize_smiles(r)
        if c:
            building_blocks.add(c)

    with open(bb_path, "wb") as f:
        pickle.dump(building_blocks, f)

    print(f"Building blocks: {len(building_blocks)} unique molecules saved")

def load_building_blocks():
    """Load set of commercially available building block SMILES."""
    bb_path = os.path.join(DATA_DIR, "building_blocks.pkl")
    if not os.path.exists(bb_path):
        return set()
    with open(bb_path, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def make_dataloader(batch_size, split, device="cpu"):
    """
    Infinite shuffled batch generator for retrosynthesis training/val data.
    Yields (inputs, targets, loss_mask, epoch).
      inputs:    (B, T)  long tensor,  T = MAX_SEQ_LEN - 1
      targets:   (B, T)  long tensor
      loss_mask: (B, T)  float tensor (1 for reactant+eos, 0 elsewhere)
    """
    assert split in ("train", "val")
    data = torch.load(os.path.join(DATA_DIR, f"{split}_data.pt"), map_location="cpu", weights_only=True)
    sequences = data["sequences"]   # (N, MAX_SEQ_LEN)
    masks = data["loss_masks"]      # (N, MAX_SEQ_LEN - 1)
    N = len(sequences)

    epoch = 1
    perm = torch.randperm(N)
    pos = 0

    while True:
        if pos + batch_size > N:
            epoch += 1
            perm = torch.randperm(N)
            pos = 0

        idx = perm[pos : pos + batch_size]
        batch_seq = sequences[idx]
        batch_mask = masks[idx]

        inputs  = batch_seq[:, :-1].to(device)
        targets = batch_seq[:, 1:].to(device)
        mask    = batch_mask.to(device)

        yield inputs, targets, mask, epoch
        pos += batch_size

# ---------------------------------------------------------------------------
# Generation utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, prefix_ids, max_new_tokens, eos_id, device):
    """
    Greedy autoregressive generation.
    prefix_ids: (B, prefix_len) tensor of token IDs.
    Returns (B, prefix_len + generated_len) tensor.
    """
    generated = prefix_ids.to(device)
    B = generated.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        if generated.size(1) >= MAX_SEQ_LEN:
            break
        logits = model(generated)            # (B, T, vocab_size)
        next_tok = logits[:, -1, :].argmax(dim=-1)  # (B,)
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
        finished |= (next_tok == eos_id)
        if finished.all():
            break

    return generated

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_retro_accuracy(model, device):
    """
    Evaluate retrosynthesis top-1 exact-match accuracy on validation set.
    Returns (accuracy, validity) as floats in [0, 1].
    """
    tokenizer = SMILESTokenizer.from_file()
    data = torch.load(os.path.join(DATA_DIR, "val_data.pt"), map_location="cpu", weights_only=True)
    sequences = data["sequences"]  # (N, MAX_SEQ_LEN)

    n_eval = min(EVAL_SAMPLES, len(sequences))
    correct = 0
    valid_count = 0

    # Use autocast if on CUDA
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    with ctx:
        for i in range(n_eval):
            seq = sequences[i]

            # Find SEP position
            sep_mask = (seq == SEP_ID)
            if not sep_mask.any():
                continue
            sep_pos = sep_mask.nonzero(as_tuple=True)[0][0].item()

            # Prefix: tokens up to and including SEP
            prefix = seq[: sep_pos + 1].unsqueeze(0)  # (1, prefix_len)

            # Ground truth reactants: tokens after SEP until EOS or PAD
            gt_ids = []
            for j in range(sep_pos + 1, len(seq)):
                tok = seq[j].item()
                if tok in (EOS_ID, PAD_ID):
                    break
                gt_ids.append(tok)
            gt_smiles = tokenizer.decode(gt_ids)
            gt_canonical = canonicalize_reaction_smiles(gt_smiles)

            # Generate
            max_new = MAX_SEQ_LEN - prefix.size(1)
            gen = generate(model, prefix, max_new, EOS_ID, device)

            # Extract predicted reactant tokens
            pred_ids = []
            for j in range(prefix.size(1), gen.size(1)):
                tok = gen[0, j].item()
                if tok in (EOS_ID, PAD_ID):
                    break
                pred_ids.append(tok)
            pred_smiles = tokenizer.decode(pred_ids)
            pred_canonical = canonicalize_reaction_smiles(pred_smiles)

            if pred_canonical is not None:
                valid_count += 1
                if pred_canonical == gt_canonical:
                    correct += 1

    accuracy = correct / n_eval if n_eval > 0 else 0.0
    validity = valid_count / n_eval if n_eval > 0 else 0.0
    return accuracy, validity

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for retrosynthesis autoresearch")
    parser.add_argument("--tiny", action="store_true", help="Use tiny subset (500 reactions) for local testing")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download, canonicalize, tokenize
    download_and_process(tiny=args.tiny)
    print()

    # Step 2: Extract building blocks
    if not args.tiny:
        extract_building_blocks()
    else:
        print("Building blocks: skipped (--tiny mode)")
    print()

    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s. Ready to train.")
