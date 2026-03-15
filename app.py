"""
RetroSynth: AI-Powered Retrosynthesis Prediction
Gradio frontend for the autoresearch-retro project.

Usage: uv run app.py
"""

import os
import pickle
from io import BytesIO

import torch
import gradio as gr
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_building_blocks = None

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from prepare import SMILESTokenizer, DATA_DIR
    from train import GPT, GPTConfig

    device = _get_device()
    _tokenizer = SMILESTokenizer.from_file()

    # Try best_model.pt first, fall back to latest_model.pt
    for path in ("best_model.pt", "latest_model.pt"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            config = GPTConfig(**checkpoint["config"])
            _model = GPT(config).to(device)
            _model.load_state_dict(checkpoint["model_state_dict"])
            _model.eval()
            print(f"Loaded model from {path} (accuracy={checkpoint.get('val_accuracy', '?')})")
            return _model, _tokenizer

    raise FileNotFoundError("No model checkpoint found. Run train.py first.")

def load_building_blocks():
    global _building_blocks
    if _building_blocks is not None:
        return _building_blocks
    from prepare import load_building_blocks as _load_bb
    _building_blocks = _load_bb()
    return _building_blocks

# ---------------------------------------------------------------------------
# Molecule visualization
# ---------------------------------------------------------------------------

def smiles_to_image(smiles, size=(300, 300)):
    """Render a SMILES string as a 2D structure image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)

def multi_smiles_to_image(smiles_list, size=(300, 300)):
    """Render multiple SMILES as a grid image."""
    mols = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        mols.append(mol)  # None is ok, Draw handles it
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, molsPerRow=min(len(mols), 4), subImgSize=size)

# ---------------------------------------------------------------------------
# Retrosynthesis prediction
# ---------------------------------------------------------------------------

def predict_single_step(product_smiles):
    """Predict reactants for a single product."""
    from prepare import (
        BOS_ID, SEP_ID, EOS_ID, MAX_SEQ_LEN,
        canonicalize_smiles, canonicalize_reaction_smiles, generate,
    )

    model, tokenizer = load_model()
    device = _get_device()

    # Canonicalize input
    canonical = canonicalize_smiles(product_smiles)
    if canonical is None:
        return None, "Invalid SMILES"

    # Encode: <bos> [product] <sep>
    product_ids = tokenizer.encode(canonical)
    prefix = [BOS_ID] + product_ids + [SEP_ID]
    prefix_tensor = torch.tensor([prefix], dtype=torch.long)

    # Generate
    max_new = MAX_SEQ_LEN - len(prefix)
    with torch.no_grad():
        generated = generate(model, prefix_tensor, max_new, EOS_ID, device)

    # Decode reactant tokens
    pred_ids = []
    for j in range(len(prefix), generated.size(1)):
        tok = generated[0, j].item()
        if tok in (EOS_ID, 0):  # EOS or PAD
            break
        pred_ids.append(tok)

    pred_smiles = tokenizer.decode(pred_ids)
    pred_canonical = canonicalize_reaction_smiles(pred_smiles)

    if pred_canonical is None:
        return pred_smiles, "Generated SMILES is invalid"

    return pred_canonical, "Valid"

def predict_multi_step(product_smiles, max_depth=5):
    """
    Recursive multi-step retrosynthesis.
    Returns a tree structure:
    {
        "smiles": str,
        "name": str or None,
        "buyable": bool,
        "children": [tree, ...] or None,
        "error": str or None,
    }
    """
    from prepare import canonicalize_smiles

    building_blocks = load_building_blocks()

    canonical = canonicalize_smiles(product_smiles)
    if canonical is None:
        return {"smiles": product_smiles, "buyable": False, "children": None, "error": "Invalid SMILES"}

    seen = set()  # cycle detection

    def _recurse(smiles, depth):
        if smiles in seen:
            return {"smiles": smiles, "buyable": False, "children": None, "error": "Cycle detected"}
        seen.add(smiles)

        # Check if commercially available
        if smiles in building_blocks:
            return {"smiles": smiles, "buyable": True, "children": None, "error": None}

        # Check depth limit
        if depth >= max_depth:
            return {"smiles": smiles, "buyable": False, "children": None, "error": "Max depth reached"}

        # Predict reactants
        pred_smiles, status = predict_single_step(smiles)
        if pred_smiles is None or "Invalid" in status or "invalid" in status:
            return {"smiles": smiles, "buyable": False, "children": None, "error": f"Prediction failed: {status}"}

        # Split into individual reactants and recurse
        fragments = pred_smiles.split(".")
        children = []
        for frag in fragments:
            frag = frag.strip()
            if frag:
                child = _recurse(frag, depth + 1)
                children.append(child)

        return {"smiles": smiles, "buyable": False, "children": children, "error": None}

    return _recurse(canonical, 0)

# ---------------------------------------------------------------------------
# Format results for display
# ---------------------------------------------------------------------------

def format_tree_as_markdown(tree, indent=0):
    """Format a retrosynthesis tree as readable markdown."""
    lines = []
    prefix = "  " * indent

    smiles = tree["smiles"]
    if tree["buyable"]:
        lines.append(f"{prefix}- **{smiles}** -- commercially available")
    elif tree["error"]:
        lines.append(f"{prefix}- **{smiles}** -- {tree['error']}")
    elif tree["children"]:
        lines.append(f"{prefix}- **{smiles}**")
        lines.append(f"{prefix}  Predicted reactants:")
        for child in tree["children"]:
            lines.extend(format_tree_as_markdown(child, indent + 2))
    else:
        lines.append(f"{prefix}- **{smiles}**")

    return lines

def tree_to_steps(tree, steps=None, step_num=None):
    """Flatten a tree into numbered synthesis steps (bottom-up)."""
    if steps is None:
        steps = []
        step_num = [0]

    if tree["children"]:
        # First recurse into children
        for child in tree["children"]:
            tree_to_steps(child, steps, step_num)

        # Then add this step
        step_num[0] += 1
        reactant_smiles = [c["smiles"] for c in tree["children"]]
        steps.append({
            "step": step_num[0],
            "product": tree["smiles"],
            "reactants": reactant_smiles,
            "buyable_reactants": [c["buyable"] for c in tree["children"]],
        })

    return steps

# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def resolve_molecule_name(name_or_smiles):
    """Try to resolve a molecule name to SMILES using pubchempy."""
    # If it looks like SMILES already (contains special chars), return as-is
    if any(c in name_or_smiles for c in "()=#[]@/\\"):
        return name_or_smiles

    try:
        import pubchempy as pcp
        results = pcp.get_compounds(name_or_smiles, "name")
        if results:
            return results[0].canonical_smiles
    except Exception:
        pass

    return name_or_smiles  # return as-is, let SMILES parsing handle errors

def predict_route(input_text, max_depth):
    """Main prediction function for Gradio."""
    if not input_text or not input_text.strip():
        return "Please enter a molecule name or SMILES string.", None, ""

    # Resolve name to SMILES
    smiles = resolve_molecule_name(input_text.strip())

    # Draw the target molecule
    target_img = smiles_to_image(smiles, size=(400, 400))
    if target_img is None:
        return f"Could not parse molecule: {smiles}", None, ""

    # Multi-step retrosynthesis
    tree = predict_multi_step(smiles, max_depth=int(max_depth))

    # Format as markdown
    md_lines = format_tree_as_markdown(tree)
    route_md = "\n".join(md_lines)

    # Get flattened steps for display
    steps = tree_to_steps(tree)
    if steps:
        step_text = []
        for s in steps:
            reactant_labels = []
            for r, b in zip(s["reactants"], s["buyable_reactants"]):
                label = f"{r} (buyable)" if b else r
                reactant_labels.append(label)
            step_text.append(f"**Step {s['step']}:** {' + '.join(reactant_labels)} --> {s['product']}")
        steps_md = "\n\n".join(step_text)
    else:
        steps_md = "No synthesis steps predicted."

    # Create reactant images for the first step
    if steps:
        last_step = steps[-1]  # the final step produces the target
        all_mols = last_step["reactants"]
        reactant_img = multi_smiles_to_image(all_mols, size=(250, 250))
    else:
        reactant_img = None

    summary = f"**Target:** `{smiles}`\n\n**Route ({len(steps)} steps):**\n\n{steps_md}\n\n---\n\n**Full tree:**\n\n{route_md}"

    return summary, target_img, reactant_img

# Demo molecule presets
DEMOS = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Adipic acid": "OC(=O)CCCCC(=O)O",
}

def set_demo(name):
    return DEMOS.get(name, "")

# Build interface
with gr.Blocks(title="RetroSynth", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RetroSynth: AI-Powered Retrosynthesis Prediction")
    gr.Markdown("Enter a molecule (SMILES or common name) to predict a synthesis route from commercial starting materials.")

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="Molecule (SMILES or name)",
                placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O or Aspirin",
                lines=1,
            )
            max_depth = gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Max retrosynthesis depth")
            predict_btn = gr.Button("Predict Route", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**Demo molecules:**")
            with gr.Row():
                for name in DEMOS:
                    gr.Button(name, size="sm").click(fn=lambda n=name: set_demo(n), outputs=input_box)

    with gr.Row():
        target_image = gr.Image(label="Target molecule", type="pil")
        reactant_image = gr.Image(label="Predicted reactants (step 1)", type="pil")

    route_output = gr.Markdown(label="Synthesis route")

    predict_btn.click(
        fn=predict_route,
        inputs=[input_box, max_depth],
        outputs=[route_output, target_image, reactant_image],
    )

    # Also show autoresearch progress if available
    if os.path.exists("progress.png"):
        gr.Markdown("---\n## Autoresearch Progress")
        gr.Image("progress.png", label="Experiment progress")

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
