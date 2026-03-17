# RetroSynth Autoresearch

**Autonomous AI-driven retrosynthesis model optimization.**

An AI agent autonomously experiments with model architecture, hyperparameters, and training strategies to improve a retrosynthesis prediction model -- predicting how to synthesize target molecules from available starting materials.

Built on the [autoresearch](https://github.com/karpathy/autoresearch) framework by Andrej Karpathy, with improvements inspired by [auto-q-research](https://github.com/ottogin/auto-q-research) for structured experiment tracking, novelty-guided exploration, and multi-step investment strategies.

## How it works

An AI coding agent (Claude, Codex, etc.) sits in a loop:

```
1. Read analysis report (training dynamics, tried configs, novelty score)
2. Formulate hypothesis (guided by exploration metrics, not just intuition)
3. Modify train.py (architecture, optimizer, hyperparameters, etc.)
4. Train the model for 5 minutes
5. Run analyze.py (log results, check invest state, update reports)
6. If accuracy improved → keep the change
   If accuracy dropped but foundational → invest (with deadline)
   If accuracy didn't improve → revert
7. Repeat forever
```

The model learns to predict **reactants from products**: given a target molecule as a SMILES string, generate the precursor molecules needed to synthesize it.

```
Input:   CC(=O)Oc1ccccc1C(=O)O          (aspirin)
Output:  CC(=O)OC(=O)C.OC(=O)c1ccccc1O  (acetic anhydride + salicylic acid)
```

The Gradio frontend provides multi-step retrosynthesis -- recursively decomposing complex molecules until all precursors are commercially available building blocks.

## Quick Start

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- For real training: NVIDIA GPU (tested on T4/A10G via AWS). CPU/MPS works for testing.

### 1. Install and prepare data

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Basic: download USPTO-50K, build tokenizer, extract building blocks (~2 min)
uv run prepare.py

# With SMILES augmentation (5x training data, recommended):
uv run prepare.py --augment 5

# With reaction class conditioning:
uv run prepare.py --augment 5 --reaction-class

# For local testing with a tiny subset (500 reactions):
uv run prepare.py --tiny

# Re-process data (e.g., after changing augmentation settings):
uv run prepare.py --force --augment 5 --reaction-class
```

### 2. Run a single training experiment

```bash
# Full 5-minute run (GPU)
uv run train.py

# Quick 30-second test (CPU/MPS)
TIME_BUDGET=30 uv run train.py
```

Output:
```
---
val_accuracy:     0.120000
val_validity:     0.850000
training_seconds: 300.1
total_seconds:    340.2
peak_vram_mb:     2048.0
total_tokens_M:   12.5
num_steps:        500
num_params_M:     2.1
depth:            4
```

### 3. Launch the autonomous agent

Point your AI coding agent at the repo with permissions disabled:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent creates a branch, establishes a baseline, then loops indefinitely -- modifying `train.py`, running experiments, analyzing results via `analyze.py`, keeping improvements (or investing in foundational changes), and logging everything.

### 4. Launch the frontend

```bash
uv run app.py
```

Opens a Gradio web UI at `http://localhost:7860` where you can:
- Input molecules (SMILES or common names like "aspirin")
- See predicted retrosynthesis routes with 2D structure drawings
- Toggle beam search for top-3 predictions with confidence scores
- View autoresearch experiment history and interactive accuracy charts

### 5. Visualize progress

Open `analysis.ipynb` to see the autoresearch progress chart, or use the **Autoresearch Progress** tab in the Gradio frontend for interactive Plotly charts.

## Architecture

```
 You (the researcher)
  |
  |  write research strategy + prepare data
  v
 program.md ──────────> AI Agent (Claude, Codex, etc.)
                           |
                           |  1. read analysis.txt (training dynamics, novelty score)
                           |  2. formulate hypothesis (guided by exploration metrics)
                           |  3. modify train.py, commit
                           v
                        train.py ──> 5-min training ──> val_accuracy
                           |                              |
                           v                              v
                      analyze.py ←─────────────────── results
                           |
                           |  log to experiments.jsonl
                           |  compute novelty score, training dynamics
                           |  check invest state (deadlines, abort thresholds)
                           |  generate fixed-size analysis.txt
                           v
                      analysis.txt ──> agent reads ──> next experiment
                           |
                        app.py <── best_model.pt ── checkpoint
                        (Gradio frontend with beam search, Plotly charts)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams.

### File structure

| File | Role | Who edits it |
|------|------|-------------|
| `prepare.py` | Data download (USPTO-50K), SMILES tokenizer, dataloader, evaluation, beam search, SMILES augmentation, reaction class conditioning | Researcher (read-only for agent) |
| `train.py` | GPT model instantiation, optimizer (AdamW), training loop, all hyperparameters, checkpoint saving | The AI agent |
| `model.py` | GPT model definition (shared by train.py and app.py) | Researcher |
| `program.md` | Agent instructions: experiment loop with analysis, invest mechanism, exploration guidance, literature search | Researcher |
| `analyze.py` | Post-experiment analysis: logging, training dynamics, novelty scores, invest state management, fixed-size reports | Researcher (run by agent, not modified) |
| `app.py` | Gradio frontend: retrosynthesis prediction with beam search, molecule visualization, experiment history, Plotly charts | Researcher |
| `analysis.ipynb` | Results visualization: progress chart, summary stats | Run after experiments |
| `terraform/` | AWS infrastructure: g4dn.xlarge GPU instance (~$0.53/hr) | Researcher |

### Key files generated during experiments

| File | Purpose | Who reads it |
|------|---------|-------------|
| `experiments.jsonl` | Append-only full experiment history with configs | Only `analyze.py` |
| `analysis.txt` | Fixed-size report (~50 lines): dynamics, tried configs, novelty, invest state | Agent (every cycle) |
| `ideas.md` | Prioritized experiment ideas (max 10, mechanically capped) | Agent |
| `invest_state.json` | Invest mechanism state (active/inactive, deadline, thresholds) | Agent + `analyze.py` |
| `results.tsv` | Legacy experiment log (6 columns) | Legacy support |
| `best_model.pt` | Best model checkpoint | `app.py`, deployment |
| `loss_curve.csv` | Per-step training loss for latest experiment | `analyze.py` |

### What's inside train.py

- **GPT model** -- decoder-only transformer with RoPE, RMS normalization, ReLU-squared activation, scaled dot-product attention (auto-dispatches to FlashAttention on supported hardware)
- **Sequence format** -- `<bos> [product SMILES] <sep> [reactant SMILES] <eos>`, with loss computed only on reactant tokens
- **Training loop** -- gradient accumulation, time-based progress tracking, LR scheduling (linear warmup + cosine warmdown), fast-fail on NaN/divergence
- **Runtime autocast test** -- verifies float16/bfloat16 works on the current GPU before training; falls back to float32 with a warning if CUBLAS errors occur
- **Outputs** -- model checkpoints (`best_model.pt`), per-step loss curve (`loss_curve.csv`), summary metrics

## Training Data & Augmentation

### USPTO-50K
- 49,015 reactions from US pharma patents
- Split: ~40K train, ~5K val, ~5K test
- 10 reaction type classes (heteroatom alkylation, acylation, C-C bond formation, etc.)

### SMILES Augmentation
The same molecule can be written as many valid SMILES strings. `prepare.py --augment N` generates N random SMILES variants per training reaction via RDKit, effectively multiplying training data with zero runtime overhead. This is a well-established technique that typically improves retrosynthesis accuracy by 5-15%.

### Reaction Class Conditioning
`prepare.py --reaction-class` extracts the 10 USPTO-50K reaction type labels and prepends class tokens (`<class_0>` through `<class_9>`) to each sequence. This gives the model a structural hint about what kind of reaction to predict.

## Agent Loop Improvements

### Structured Analysis (`analyze.py`)
After each experiment, `analyze.py` produces a fixed-size `analysis.txt` report with:
- Training dynamics (convergence detection, end-of-run slope, trend)
- Tried-configurations summary (one line per hyperparameter, prevents retrying)
- Config-space novelty score (deterministic L2 distance, flags low-diversity exploration)
- Diminishing returns detection (flags tapped-out hyperparameter dimensions)
- Invest mechanism state (deadline tracking, abort thresholds)

### Invest Mechanism
Beyond keep/discard, the agent can mark an experiment as **invest** when accuracy dropped but the change is believed to be foundational (e.g., SMILES augmentation needs follow-up model scaling). Invests have deadlines and abort thresholds, enforced mechanically by `analyze.py`.

### Novelty-Guided Exploration
Each experiment's config is projected into a normalized hyperparameter space, and its distance from all prior experiments is computed. Low novelty triggers a suggestion to explore a different direction, preventing the agent from making tiny variations in one corner of the search space.

## Inference

### Beam Search
`prepare.py` includes both greedy and beam search generation. Beam search (width 10) explores multiple candidate predictions in parallel, typically adding +5-10% top-1 accuracy. The frontend supports toggling beam search for top-3 predictions with log-probability scores.

### Multi-Step Retrosynthesis
The model predicts one step; the frontend recursively applies it, stopping when reactants are commercially available (checked against building blocks extracted from training data) or when max depth is reached.

## Deployment Options

### AWS (recommended for autonomous agent runs)

Terraform config provisions a `g4dn.xlarge` (T4 16GB GPU, ~$0.53/hr):

```bash
cd terraform
terraform init
terraform apply -var="ssh_key_name=your-key"
# SSH in, start Claude Code in tmux, let it run overnight
```

### Google Colab (free, for testing and demos)

Open `colab.ipynb` in Colab, select T4 GPU runtime, run cells top to bottom. Free but limited: no autonomous agent, sessions timeout.

### Hugging Face Spaces (free, for persistent demos)

Deploy the Gradio frontend to HF Spaces for a permanent demo:
1. Train on EC2 (agent loop)
2. Copy `best_model.pt` locally
3. Push repo + model to a HF Space
4. Shut down EC2

### Local (MacBook, for development)

```bash
uv run prepare.py --tiny
TIME_BUDGET=30 uv run train.py
uv run app.py
```

## GPU Compatibility

The training script automatically detects GPU capability and selects the appropriate precision:
- **Compute >= 8.0** (A10G, A100, L4): bfloat16
- **Compute < 8.0** (T4): float16
- **CUBLAS failure**: automatic fallback to float32 with warning

PyTorch is pinned to 2.4-2.5 with CUDA 12.4 for maximum T4 compatibility.

## Attribution

This project is an unofficial fork/adaptation of [autoresearch](https://github.com/karpathy/autoresearch) by [Andrej Karpathy](https://github.com/karpathy). The agent loop improvements (structured analysis, novelty scores, invest mechanism) are inspired by [auto-q-research](https://github.com/ottogin/auto-q-research).

## License

MIT
