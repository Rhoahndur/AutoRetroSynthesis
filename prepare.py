"""
One-time data preparation for retrosynthesis autoresearch experiments.
Downloads USPTO-50K, builds SMILES tokenizer, extracts building blocks.

Usage:
    uv run prepare.py                          # full prep (canonical only)
    uv run prepare.py --augment 5              # 5x SMILES augmentation (training only)
    uv run prepare.py --reaction-class         # add reaction class tokens
    uv run prepare.py --augment 5 --reaction-class  # both
    uv run prepare.py --tiny                   # tiny subset for local testing
    uv run prepare.py --force                  # re-process even if data exists

Data stored in ~/.cache/autoresearch-retro/
"""

import os
import re
import sys
import json
import time
import random
import pickle
import argparse
from contextlib import nullcontext
from collections import Counter

import torch

# Suppress all RDKit C++ warnings globally (model generates lots of invalid SMILES)
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 256  # context length (reactions are short)
TIME_BUDGET = 300  # training time budget in seconds (5 minutes)
EVAL_SAMPLES = 500  # number of val samples for accuracy evaluation
NUM_REACTION_CLASSES = 10  # USPTO-50K has 10 reaction types

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
# Class tokens <class_0>..<class_9> are always in vocab (IDs 4-13).
# Models trained without --reaction-class simply never see these tokens.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
CLASS_ID_OFFSET = 4  # <class_0> = 4, <class_1> = 5, ..., <class_9> = 13
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>"] + [
    f"<class_{i}>" for i in range(NUM_REACTION_CLASSES)
]

# ---------------------------------------------------------------------------
# SMILES Canonicalization (RDKit)
# ---------------------------------------------------------------------------


def _get_chem():
    """Lazy import RDKit Chem module with all warnings disabled."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
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
# SMILES Augmentation (random non-canonical SMILES)
# ---------------------------------------------------------------------------


def randomize_smiles(smiles):
    """Generate a random valid SMILES string for the same molecule.
    Returns original SMILES if randomization fails."""
    Chem = _get_chem()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return smiles


def randomize_reaction_smiles(smiles):
    """Randomize multi-fragment SMILES. Each fragment gets a random SMILES
    representation and fragment order is shuffled."""
    fragments = smiles.split(".")
    randomized = [randomize_smiles(frag) for frag in fragments]
    random.shuffle(randomized)
    return ".".join(randomized)


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
            json.dump(
                {
                    "token2id": self.token2id,
                    "id2token": {str(k): v for k, v in self.id2token.items()},
                },
                f,
                indent=2,
            )

    @classmethod
    def build_from_reactions(cls, reactions):
        """Build vocabulary from list of (product_smiles, reactant_smiles, ...) tuples.
        Extra elements (e.g. class_id) are ignored for vocab building."""
        token_set = set()
        for rxn in reactions:
            product, reactants = rxn[0], rxn[1]
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
        """Tokenize SMILES string -> list of int IDs."""
        return [
            self.token2id[t]
            for t in SMILES_PATTERN.findall(smiles)
            if t in self.token2id
        ]

    def decode(self, ids):
        """Decode list of int IDs -> SMILES string.
        Skips all special tokens (any token starting with '<')."""
        return "".join(
            self.id2token[i]
            for i in ids
            if i in self.id2token and not self.id2token[i].startswith("<")
        )

    def encode_reaction(self, product_smiles, reactant_smiles, class_id=None):
        """Encode a reaction: <bos> [<class_N>] [product] <sep> [reactants] <eos>
        class_id: optional int 0-9 for reaction class conditioning."""
        prefix = [BOS_ID]
        if class_id is not None:
            prefix.append(CLASS_ID_OFFSET + class_id)
        return (
            prefix
            + self.encode(product_smiles)
            + [SEP_ID]
            + self.encode(reactant_smiles)
            + [EOS_ID]
        )

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


def _find_class_column(row):
    """Find reaction class label in a dataset row. Returns int 0-9 or None."""
    for key in ("class", "reaction_type", "rxn_class", "reaction_class", "Class"):
        if key in row:
            val = row[key]
            if isinstance(val, (int, float)):
                class_id = int(val)
                # USPTO-50K classes are 1-10, convert to 0-9
                if 1 <= class_id <= NUM_REACTION_CLASSES:
                    return class_id - 1
                elif 0 <= class_id < NUM_REACTION_CLASSES:
                    return class_id
            elif isinstance(val, str) and val.isdigit():
                return _find_class_column({key: int(val)})
    return None


def download_and_process(tiny=False, augment=0, reaction_class=False, force=False):
    """Download USPTO-50K, canonicalize, tokenize, and save.

    Args:
        tiny: Use only 500 reactions for local testing.
        augment: Number of augmented SMILES copies per training reaction (0 = canonical only).
        reaction_class: If True, extract reaction class labels and prepend class tokens.
        force: Re-process even if data files already exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if already done
    if not force and all(
        os.path.exists(os.path.join(DATA_DIR, f"{s}_data.pt"))
        for s in ("train", "val", "test")
    ):
        print("Data: already processed (use --force to re-process)")
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
    all_reactions = {}  # split_name -> list of (product, reactants, class_id_or_None)

    for split_name, hf_split in split_map.items():
        reactions = []
        skipped = 0
        class_found = 0
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
                class_id = _find_class_column(row) if reaction_class else None
                if reaction_class and class_id is not None:
                    class_found += 1
                reactions.append((product, reactants, class_id))
            else:
                skipped += 1

        if tiny:
            reactions = reactions[:500]

        all_reactions[split_name] = reactions
        class_msg = f", {class_found} with class labels" if reaction_class else ""
        print(f"Data: {split_name}: {len(reactions)} reactions ({skipped} skipped{class_msg})")

    # Save raw reactions as JSON for future flexibility
    raw_path = os.path.join(DATA_DIR, "raw_reactions.json")
    raw_data = {}
    for split_name, reactions in all_reactions.items():
        raw_data[split_name] = [
            {"product": p, "reactants": r, "class_id": c}
            for p, r, c in reactions
        ]
    with open(raw_path, "w") as f:
        json.dump(raw_data, f)
    print(f"Data: saved raw reactions -> {raw_path}")

    # Build tokenizer from training data (canonical only -- augmented SMILES use same tokens)
    print("Tokenizer: building vocabulary...")
    tokenizer = SMILESTokenizer.build_from_reactions(all_reactions["train"])
    tokenizer.save()
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Tokenize and save each split
    for split_name, reactions in all_reactions.items():
        sequences = []
        skipped = 0

        # For training data, generate augmented copies
        is_train = split_name == "train"
        aug_factor = augment if is_train else 0

        for product, reactants, class_id in reactions:
            class_arg = class_id if reaction_class else None

            # Always include the canonical version
            seq = tokenizer.encode_reaction(product, reactants, class_id=class_arg)
            if len(seq) > MAX_SEQ_LEN:
                skipped += 1
                continue
            padded = seq + [PAD_ID] * (MAX_SEQ_LEN - len(seq))
            sequences.append(padded)

            # Generate augmented copies (training only)
            for _ in range(aug_factor):
                aug_product = randomize_smiles(product)
                aug_reactants = randomize_reaction_smiles(reactants)
                aug_seq = tokenizer.encode_reaction(aug_product, aug_reactants, class_id=class_arg)
                if len(aug_seq) <= MAX_SEQ_LEN:
                    aug_padded = aug_seq + [PAD_ID] * (MAX_SEQ_LEN - len(aug_seq))
                    sequences.append(aug_padded)

        if skipped > 0:
            print(
                f"Data: {split_name}: {skipped} reactions exceeded MAX_SEQ_LEN={MAX_SEQ_LEN}"
            )

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

        aug_msg = f" (includes {aug_factor}x augmentation)" if aug_factor > 0 else ""
        print(f"Data: saved {split_name} ({len(sequences)} sequences{aug_msg}) -> {path}")


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

    # Try loading from raw_reactions.json first (avoids re-downloading)
    raw_path = os.path.join(DATA_DIR, "raw_reactions.json")
    reactant_counts = Counter()

    if os.path.exists(raw_path):
        with open(raw_path) as f:
            raw_data = json.load(f)
        for rxn in raw_data.get("train", []):
            for frag in rxn["reactants"].split("."):
                c = canonicalize_smiles(frag.strip())
                if c:
                    reactant_counts[c] += 1
    else:
        # Fallback: re-download dataset
        from datasets import load_dataset

        ds = None
        for name in ("pingzhili/uspto-50k", "sagawa/USPTO-50k"):
            try:
                ds = load_dataset(name)
                break
            except Exception:
                pass

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
    data = torch.load(
        os.path.join(DATA_DIR, f"{split}_data.pt"),
        map_location="cpu",
        weights_only=True,
    )
    sequences = data["sequences"]  # (N, MAX_SEQ_LEN)
    masks = data["loss_masks"]  # (N, MAX_SEQ_LEN - 1)
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

        inputs = batch_seq[:, :-1].to(device)
        targets = batch_seq[:, 1:].to(device)
        mask = batch_mask.to(device)

        yield inputs, targets, mask, epoch
        pos += batch_size


# ---------------------------------------------------------------------------
# Generation utilities
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
        logits = model(generated)  # (B, T, vocab_size)
        next_tok = logits[:, -1, :].argmax(dim=-1)  # (B,)
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok == eos_id
        if finished.all():
            break

    return generated


@torch.no_grad()
def generate_beam(model, prefix_ids, max_new_tokens, eos_id, device, beam_width=10):
    """
    Beam search autoregressive generation.
    prefix_ids: (1, prefix_len) tensor of token IDs (batch size must be 1).
    Returns list of (sequence_tensor, log_prob) tuples, sorted by log_prob descending.
    """
    assert prefix_ids.size(0) == 1, "Beam search requires batch size 1"
    prefix = prefix_ids.to(device)
    prefix_len = prefix.size(1)

    # Each beam: (sequence, cumulative_log_prob, finished)
    beams = [(prefix, 0.0, False)]

    for step in range(max_new_tokens):
        if all(b[2] for b in beams):
            break

        candidates = []
        for seq, log_prob, finished in beams:
            if finished or seq.size(1) >= MAX_SEQ_LEN:
                candidates.append((seq, log_prob, True))
                continue

            logits = model(seq)  # (1, T, vocab_size)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)  # (vocab_size,)

            # Take top beam_width tokens
            topk_log_probs, topk_ids = log_probs.topk(beam_width)

            for i in range(beam_width):
                tok_id = topk_ids[i].item()
                tok_log_prob = topk_log_probs[i].item()
                new_seq = torch.cat([seq, topk_ids[i].view(1, 1)], dim=1)
                new_log_prob = log_prob + tok_log_prob
                new_finished = tok_id == eos_id
                candidates.append((new_seq, new_log_prob, new_finished))

        # Keep top beam_width candidates by cumulative log prob
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    # Return sorted results
    results = []
    for seq, log_prob, _ in beams:
        results.append((seq, log_prob))
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_retro_accuracy(model, device, use_beam=False, beam_width=10):
    """
    Evaluate retrosynthesis accuracy on validation set.
    Returns (accuracy, validity) as floats in [0, 1] when use_beam=False.
    Returns (top1_acc, top3_acc, top5_acc, top10_acc, validity) when use_beam=True.
    """
    # Fully suppress RDKit C++ warnings during eval (model outputs lots of invalid SMILES)
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    tokenizer = SMILESTokenizer.from_file()
    data = torch.load(
        os.path.join(DATA_DIR, "val_data.pt"), map_location="cpu", weights_only=True
    )
    sequences = data["sequences"]  # (N, MAX_SEQ_LEN)

    n_eval = min(EVAL_SAMPLES, len(sequences))
    correct = 0
    valid_count = 0
    # For beam search: track top-K accuracy
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0

    # Use autocast if on CUDA
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = nullcontext()

    with ctx:
        for i in range(n_eval):
            seq = sequences[i]

            # Find SEP position
            sep_mask = seq == SEP_ID
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

            max_new = MAX_SEQ_LEN - prefix.size(1)

            if use_beam:
                # Beam search: get top-K predictions
                beam_results = generate_beam(model, prefix, max_new, EOS_ID, device, beam_width=beam_width)
                found_at_k = None
                first_valid = False
                for rank, (gen_seq, _log_prob) in enumerate(beam_results):
                    pred_ids = []
                    for j in range(prefix.size(1), gen_seq.size(1)):
                        tok = gen_seq[0, j].item()
                        if tok in (EOS_ID, PAD_ID):
                            break
                        pred_ids.append(tok)
                    pred_smiles = tokenizer.decode(pred_ids)
                    pred_canonical = canonicalize_reaction_smiles(pred_smiles)

                    if rank == 0 and pred_canonical is not None:
                        first_valid = True

                    if pred_canonical is not None and pred_canonical == gt_canonical:
                        if found_at_k is None:
                            found_at_k = rank
                        break

                if first_valid:
                    valid_count += 1
                if found_at_k is not None:
                    if found_at_k < 1:
                        correct += 1
                    if found_at_k < 3:
                        top3_correct += 1
                    if found_at_k < 5:
                        top5_correct += 1
                    if found_at_k < 10:
                        top10_correct += 1
            else:
                # Greedy: single prediction
                gen = generate(model, prefix, max_new, EOS_ID, device)

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

            # Progress indicator every 100 examples
            if (i + 1) % 100 == 0:
                print(
                    f"\r  eval: {i + 1}/{n_eval} (acc={correct}/{i + 1}, valid={valid_count}/{i + 1})",
                    end="",
                    flush=True,
                )

    if n_eval >= 100:
        print()  # newline after progress

    # Restore RDKit logging
    RDLogger.EnableLog("rdApp.*")

    accuracy = correct / n_eval if n_eval > 0 else 0.0
    validity = valid_count / n_eval if n_eval > 0 else 0.0

    if use_beam:
        top3_acc = top3_correct / n_eval if n_eval > 0 else 0.0
        top5_acc = top5_correct / n_eval if n_eval > 0 else 0.0
        top10_acc = top10_correct / n_eval if n_eval > 0 else 0.0
        return accuracy, top3_acc, top5_acc, top10_acc, validity

    return accuracy, validity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for retrosynthesis autoresearch"
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny subset (500 reactions) for local testing",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        metavar="N",
        help="Generate N augmented SMILES copies per training reaction (default: 0 = canonical only)",
    )
    parser.add_argument(
        "--reaction-class",
        action="store_true",
        help="Extract reaction class labels and prepend class tokens to sequences",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process data even if it already exists",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"Cache directory: {CACHE_DIR}")
    if args.augment > 0:
        print(f"SMILES augmentation: {args.augment}x per training reaction")
    if args.reaction_class:
        print("Reaction class conditioning: enabled")
    print()

    # Step 1: Download, canonicalize, tokenize (with optional augmentation + classes)
    download_and_process(
        tiny=args.tiny,
        augment=args.augment,
        reaction_class=args.reaction_class,
        force=args.force,
    )
    print()

    # Step 2: Extract building blocks
    if not args.tiny:
        extract_building_blocks()
    else:
        print("Building blocks: skipped (--tiny mode)")
    print()

    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s. Ready to train.")
