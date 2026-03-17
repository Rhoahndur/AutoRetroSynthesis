"""
RetroSynth: AI-Powered Retrosynthesis Prediction
Gradio frontend for the autoresearch-retro project.

Usage: uv run app.py
"""

import os
import json
import csv

os.environ.pop("MPLBACKEND", None)  # fix Colab matplotlib backend conflict with Gradio

import numpy as np
import torch
import gradio as gr
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
_model_info = {}


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    global _model, _tokenizer, _model_info
    if _model is not None:
        return _model, _tokenizer

    from prepare import SMILESTokenizer
    from model import GPT, GPTConfig

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
            _model_info = {
                "path": path,
                "accuracy": checkpoint.get("val_accuracy", None),
                "validity": checkpoint.get("val_validity", None),
                "config": checkpoint.get("config", {}),
                "modified": os.path.getmtime(path),
            }
            print(
                f"Loaded model from {path} (accuracy={_model_info['accuracy']})"
            )
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
    """Render a SMILES string as a 2D structure image, returned as numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    return np.array(img)


# ---------------------------------------------------------------------------
# Retrosynthesis prediction (with beam search support)
# ---------------------------------------------------------------------------


def predict_single_step(product_smiles, use_beam=False, beam_width=10):
    """Predict reactants for a single product.
    Returns (pred_smiles, status) for greedy mode.
    Returns ([(pred_smiles, log_prob, status), ...]) for beam mode.
    """
    from prepare import (
        BOS_ID, SEP_ID, EOS_ID, PAD_ID, MAX_SEQ_LEN,
        canonicalize_smiles, canonicalize_reaction_smiles,
        generate, generate_beam,
    )

    model, tokenizer = load_model()
    device = _get_device()

    canonical = canonicalize_smiles(product_smiles)
    if canonical is None:
        if use_beam:
            return [(None, 0.0, "Invalid SMILES")]
        return None, "Invalid SMILES"

    product_ids = tokenizer.encode(canonical)
    prefix = [BOS_ID] + product_ids + [SEP_ID]
    prefix_tensor = torch.tensor([prefix], dtype=torch.long)
    max_new = MAX_SEQ_LEN - len(prefix)

    if not use_beam:
        with torch.no_grad():
            generated = generate(model, prefix_tensor, max_new, EOS_ID, device)
        pred_ids = []
        for j in range(len(prefix), generated.size(1)):
            tok = generated[0, j].item()
            if tok in (EOS_ID, PAD_ID):
                break
            pred_ids.append(tok)
        pred_smiles = tokenizer.decode(pred_ids)
        pred_canonical = canonicalize_reaction_smiles(pred_smiles)
        if pred_canonical is None:
            return pred_smiles, "Generated SMILES is invalid"
        return pred_canonical, "Valid"

    # Beam search mode
    with torch.no_grad():
        beam_results = generate_beam(model, prefix_tensor, max_new, EOS_ID, device, beam_width=beam_width)

    predictions = []
    seen = set()
    for seq, log_prob in beam_results:
        pred_ids = []
        for j in range(len(prefix), seq.size(1)):
            tok = seq[0, j].item()
            if tok in (EOS_ID, PAD_ID):
                break
            pred_ids.append(tok)
        pred_smiles = tokenizer.decode(pred_ids)
        pred_canonical = canonicalize_reaction_smiles(pred_smiles)
        if pred_canonical is not None and pred_canonical not in seen:
            seen.add(pred_canonical)
            predictions.append((pred_canonical, log_prob, "Valid"))
        elif pred_canonical is None and pred_smiles not in seen:
            seen.add(pred_smiles)
            predictions.append((pred_smiles, log_prob, "Invalid SMILES"))

    if not predictions:
        predictions = [(None, 0.0, "No valid predictions")]

    return predictions


def predict_multi_step(product_smiles, max_depth=5):
    """Recursive multi-step retrosynthesis."""
    from prepare import canonicalize_smiles

    building_blocks = load_building_blocks()

    canonical = canonicalize_smiles(product_smiles)
    if canonical is None:
        return {"smiles": product_smiles, "buyable": False, "children": None, "error": "Invalid SMILES"}

    seen = set()

    def _recurse(smiles, depth):
        if smiles in seen:
            return {"smiles": smiles, "buyable": False, "children": None, "error": "Cycle detected"}
        seen.add(smiles)

        if smiles in building_blocks:
            return {"smiles": smiles, "buyable": True, "children": None, "error": None}

        if depth >= max_depth:
            return {"smiles": smiles, "buyable": False, "children": None, "error": "Max depth reached"}

        pred_smiles, status = predict_single_step(smiles)
        if pred_smiles is None or "Invalid" in status or "invalid" in status:
            return {"smiles": smiles, "buyable": False, "children": None, "error": f"Prediction failed: {status}"}

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


def tree_to_steps(tree, steps=None, step_num=None):
    """Flatten a tree into numbered synthesis steps (bottom-up)."""
    if steps is None:
        steps = []
        step_num = [0]

    if tree["children"]:
        for child in tree["children"]:
            tree_to_steps(child, steps, step_num)
        step_num[0] += 1
        reactant_smiles = [c["smiles"] for c in tree["children"]]
        steps.append({
            "step": step_num[0],
            "product": tree["smiles"],
            "reactants": reactant_smiles,
            "buyable_reactants": [c["buyable"] for c in tree["children"]],
        })

    return steps


def resolve_molecule_name(name_or_smiles):
    """Try to resolve a molecule name to SMILES using pubchempy."""
    if any(c in name_or_smiles for c in "()=#[]@/\\"):
        return name_or_smiles
    try:
        import pubchempy as pcp
        results = pcp.get_compounds(name_or_smiles, "name")
        if results:
            return results[0].canonical_smiles
    except Exception:
        pass
    return name_or_smiles


def _smiles_to_name(smiles):
    """Try to get a common name for a SMILES string."""
    try:
        import pubchempy as pcp
        results = pcp.get_compounds(smiles, "smiles")
        if results and results[0].iupac_name:
            name = results[0].iupac_name
            if results[0].synonyms:
                name = results[0].synonyms[0]
            return name
    except Exception:
        pass
    return None


def _render_step_image(product_smiles, reactant_smiles_list, step_num):
    """Render a single retrosynthesis step as one image with product and reactants."""
    mols = []
    legends = []

    prod_mol = Chem.MolFromSmiles(product_smiles)
    if prod_mol:
        mols.append(prod_mol)
        name = _smiles_to_name(product_smiles)
        legends.append("Product" if not name else f"Product: {name}")

    for i, r in enumerate(reactant_smiles_list):
        mol = Chem.MolFromSmiles(r)
        if mol:
            mols.append(mol)
            name = _smiles_to_name(r)
            label = f"Reactant {i + 1}" if not name else f"Reactant: {name}"
            legends.append(label)

    if not mols:
        return None

    img = Draw.MolsToGridImage(
        mols, molsPerRow=min(len(mols), 4), subImgSize=(300, 300), legends=legends
    )
    return np.array(img)


# ---------------------------------------------------------------------------
# Experiment history & charts
# ---------------------------------------------------------------------------


def load_experiment_history():
    """Load experiment history from experiments.jsonl or results.tsv."""
    experiments = []

    # Try experiments.jsonl first (new format)
    jsonl_path = os.path.join(os.path.dirname(__file__), "experiments.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    experiments.append(json.loads(line))
        return experiments

    # Fall back to results.tsv (old format)
    tsv_path = os.path.join(os.path.dirname(__file__), "results.tsv")
    if os.path.exists(tsv_path):
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                experiments.append({
                    "commit": row.get("commit", ""),
                    "val_accuracy": float(row.get("val_accuracy", 0)),
                    "val_validity": float(row.get("val_validity", 0)),
                    "memory_gb": float(row.get("memory_gb", 0)),
                    "status": row.get("status", ""),
                    "description": row.get("description", ""),
                })
        return experiments

    return []


def make_accuracy_chart():
    """Create an interactive accuracy-over-experiments chart."""
    experiments = load_experiment_history()
    if not experiments:
        return None

    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    exp_nums = list(range(1, len(experiments) + 1))
    accuracies = [e.get("val_accuracy", 0) for e in experiments]
    statuses = [e.get("status", "discard") for e in experiments]
    descriptions = [e.get("description", "") for e in experiments]

    # Running best (cummax)
    running_best = []
    best_so_far = 0
    for acc in accuracies:
        best_so_far = max(best_so_far, acc)
        running_best.append(best_so_far)

    # Color by status
    colors = []
    for s in statuses:
        if s == "keep":
            colors.append("#22c55e")  # green
        elif s == "invest":
            colors.append("#f59e0b")  # amber
        elif s == "crash":
            colors.append("#ef4444")  # red
        else:
            colors.append("#9ca3af")  # gray

    fig = go.Figure()

    # Running best line
    fig.add_trace(go.Scatter(
        x=exp_nums, y=running_best,
        mode="lines",
        name="Best accuracy",
        line=dict(color="#22c55e", width=2, dash="dot"),
    ))

    # Individual experiments
    fig.add_trace(go.Scatter(
        x=exp_nums, y=accuracies,
        mode="markers",
        name="Experiments",
        marker=dict(color=colors, size=10, line=dict(width=1, color="white")),
        text=[f"{s}: {d}" for s, d in zip(statuses, descriptions)],
        hovertemplate="Experiment %{x}<br>Accuracy: %{y:.4f}<br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        title="Autoresearch Progress",
        xaxis_title="Experiment #",
        yaxis_title="Validation Accuracy",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
    )

    return fig


def make_experiment_table_md():
    """Create markdown table of experiment history."""
    experiments = load_experiment_history()
    if not experiments:
        return "No experiment history available."

    lines = ["| # | Commit | Accuracy | Validity | Status | Description |",
             "|---|--------|----------|----------|--------|-------------|"]

    for i, exp in enumerate(experiments):
        status = exp.get("status", "?")
        status_icon = {"keep": "kept", "discard": "discarded", "invest": "invested", "crash": "crashed"}.get(status, status)
        lines.append(
            f"| {i + 1} | `{exp.get('commit', '?')[:7]}` | "
            f"{exp.get('val_accuracy', 0):.4f} | "
            f"{exp.get('val_validity', 0):.4f} | "
            f"{status_icon} | "
            f"{exp.get('description', '')[:50]} |"
        )

    return "\n".join(lines)


def get_model_info_md():
    """Get model info badge as markdown."""
    if not _model_info:
        try:
            load_model()
        except FileNotFoundError:
            return "No model loaded."

    info = _model_info
    acc = info.get("accuracy")
    acc_str = f"{acc:.1%}" if acc is not None else "?"
    config = info.get("config", {})
    params_str = ""
    if config:
        n_layer = config.get("n_layer", "?")
        n_embd = config.get("n_embd", "?")
        params_str = f" | {n_layer} layers, {n_embd} dim"

    import time
    mod_time = info.get("modified")
    if mod_time:
        date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mod_time))
    else:
        date_str = "?"

    return f"**Model accuracy: {acc_str}**{params_str} | Last updated: {date_str}"


# ---------------------------------------------------------------------------
# Main prediction handler
# ---------------------------------------------------------------------------


def predict_route(input_text, max_depth, use_beam_search):
    """Main prediction function for Gradio."""
    if not input_text or not input_text.strip():
        return "Please enter a molecule name or SMILES string.", None, None

    smiles = resolve_molecule_name(input_text.strip())
    target_img = smiles_to_image(smiles, size=(400, 400))
    if target_img is None:
        return f"Could not parse molecule: `{smiles}`", None, None

    target_name = _smiles_to_name(smiles)
    target_label = f"**{target_name}**" if target_name else ""

    parts = [f"## Target: {target_label}", f"`{smiles}`", ""]

    # Show beam search top-3 predictions if enabled
    if use_beam_search:
        beam_preds = predict_single_step(smiles, use_beam=True, beam_width=10)
        parts.append("### Top Predictions (beam search)")
        parts.append("")
        for rank, (pred, log_prob, status) in enumerate(beam_preds[:3]):
            confidence = f"(log-prob: {log_prob:.2f})"
            if status == "Valid":
                parts.append(f"{rank + 1}. `{pred}` {confidence}")
            else:
                parts.append(f"{rank + 1}. ~~`{pred}`~~ {confidence} -- {status}")
        parts.append("")

    # Multi-step retrosynthesis
    tree = predict_multi_step(smiles, max_depth=int(max_depth))
    steps = tree_to_steps(tree)

    if not steps:
        error = tree.get("error", "Unknown error")
        parts.append(f"**Could not predict a route:** {error}")
        return "\n".join(parts), target_img, None

    parts.append(f"### Synthesis Route ({len(steps)} steps)")
    parts.append("")

    for s in steps:
        parts.append(f"**Step {s['step']}:**")
        reactant_parts = []
        for r, b in zip(s["reactants"], s["buyable_reactants"]):
            rname = _smiles_to_name(r)
            if rname:
                label = f"**{rname}**" if not b else f"**{rname}** (buyable)"
            else:
                label = f"`{r}`" if not b else f"`{r}` (buyable)"
            reactant_parts.append(label)
        parts.append(" + ".join(reactant_parts))

        pname = _smiles_to_name(s["product"])
        prod_label = f"**{pname}**" if pname else f"`{s['product']}`"
        parts.append(f"  --> {prod_label}")
        parts.append("")

    summary = "\n".join(parts)

    last_step = steps[-1]
    reactant_img = _render_step_image(
        last_step["product"], last_step["reactants"], len(steps)
    )

    return summary, target_img, reactant_img


# ---------------------------------------------------------------------------
# Demo molecule presets
# ---------------------------------------------------------------------------

DEMOS = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Adipic acid": "OC(=O)CCCCC(=O)O",
}


def set_demo(name):
    return DEMOS.get(name, "")


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="RetroSynth", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RetroSynth: AI-Powered Retrosynthesis Prediction")
    gr.Markdown(
        "Enter a molecule (SMILES or common name) to predict a synthesis route "
        "from commercial starting materials. Powered by a GPT model trained on "
        "USPTO-50K via autonomous AI experimentation."
    )

    # Model info badge
    try:
        model_info = get_model_info_md()
    except Exception:
        model_info = ""
    if model_info:
        gr.Markdown(model_info)

    # --- Prediction tab ---
    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column(scale=2):
                input_box = gr.Textbox(
                    label="Molecule (SMILES or name)",
                    placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O or Aspirin",
                    lines=1,
                )
                with gr.Row():
                    max_depth = gr.Slider(
                        minimum=1, maximum=8, value=3, step=1,
                        label="Max retrosynthesis depth",
                    )
                    beam_toggle = gr.Checkbox(
                        label="Use beam search (top-3 predictions)",
                        value=False,
                    )
                predict_btn = gr.Button("Predict Route", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("**Demo molecules:**")
                with gr.Row():
                    for name in DEMOS:
                        gr.Button(name, size="sm").click(
                            fn=lambda n=name: set_demo(n), outputs=input_box
                        )

        with gr.Row():
            target_image = gr.Image(label="Target molecule", type="numpy")
            reactant_image = gr.Image(label="Predicted reactants", type="numpy")

        route_output = gr.Markdown(label="Synthesis route")

        predict_btn.click(
            fn=predict_route,
            inputs=[input_box, max_depth, beam_toggle],
            outputs=[route_output, target_image, reactant_image],
        )

    # --- Autoresearch tab ---
    with gr.Tab("Autoresearch Progress"):
        gr.Markdown("## Experiment History")
        gr.Markdown(
            "The model is improved via autonomous AI experimentation. "
            "Each experiment modifies the training code, trains for 5 minutes, "
            "and the change is kept only if accuracy improves."
        )

        # Accuracy chart
        chart = make_accuracy_chart()
        if chart is not None:
            gr.Plot(value=chart, label="Accuracy over experiments")
        elif os.path.exists("progress.png"):
            gr.Image("progress.png", label="Experiment progress")

        # Experiment history table
        history_md = make_experiment_table_md()
        gr.Markdown(history_md)

        # Loss curve (if available)
        if os.path.exists("loss_curve.csv"):
            gr.Markdown("### Latest Training Loss Curve")
            try:
                import plotly.graph_objects as go

                steps_data, loss_data = [], []
                with open("loss_curve.csv") as f:
                    reader = csv.reader(f)
                    next(reader)  # skip header
                    for row in reader:
                        if len(row) >= 2:
                            steps_data.append(int(row[0]))
                            loss_data.append(float(row[1]))

                if steps_data:
                    loss_fig = go.Figure()
                    loss_fig.add_trace(go.Scatter(
                        x=steps_data, y=loss_data,
                        mode="lines", name="Training loss",
                        line=dict(color="#3b82f6", width=1.5),
                    ))
                    loss_fig.update_layout(
                        xaxis_title="Step",
                        yaxis_title="Loss",
                        template="plotly_white",
                        height=300,
                        margin=dict(l=60, r=20, t=20, b=40),
                    )
                    gr.Plot(value=loss_fig, label="Loss curve")
            except Exception:
                pass

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
