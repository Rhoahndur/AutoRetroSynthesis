# RetroSynth Autoresearch

**Autonomous AI-driven retrosynthesis model optimization.**

An AI agent autonomously experiments with model architecture, hyperparameters, and training strategies to improve a retrosynthesis prediction model -- predicting how to synthesize target molecules from available starting materials.

Built on the [autoresearch](https://github.com/karpathy/autoresearch) framework by Andrej Karpathy, which introduced the idea of letting AI agents run ML experiments autonomously in a keep/discard loop. This project adapts that framework from language model pretraining to **chemical retrosynthesis prediction** on the [USPTO-50K](https://huggingface.co/datasets/pingzhili/uspto-50k) dataset, demonstrating that the autonomous experimentation pattern generalizes beyond text.

## How it works

An AI coding agent (Claude, Codex, etc.) sits in a loop:

```
1. Modify train.py (architecture, optimizer, hyperparameters, etc.)
2. Train the model for 5 minutes
3. Evaluate retrosynthesis accuracy on validation set
4. If accuracy improved → keep the change
   If accuracy didn't improve → revert
5. Log results to results.tsv
6. Repeat forever
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

# Download USPTO-50K, build tokenizer, extract building blocks (~2 min)
uv run prepare.py

# For local testing with a tiny subset (500 reactions):
uv run prepare.py --tiny
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

The agent creates a branch, establishes a baseline, then loops indefinitely -- modifying `train.py`, running experiments, keeping improvements, and logging everything to `results.tsv`.

### 4. Launch the frontend

```bash
uv run app.py
```

Opens a Gradio web UI where you can input molecules (SMILES or common names like "aspirin") and see predicted retrosynthesis routes with 2D structure drawings.

### 5. Visualize progress

Open `analysis.ipynb` to see the autoresearch progress chart -- validation accuracy climbing over experiments, with each kept improvement annotated.

## Architecture

```
 You (the researcher)
  |
  |  write research strategy
  v
 program.md ──────────> AI Agent (Claude, Codex, etc.)
                           |
                           |  modify code, commit, run, evaluate
                           v
                        train.py ──> 5-min training ──> val_accuracy
                           |                              |
                           |  improved? ──> keep commit   |
                           |  worse?    ──> git reset     |
                           v                              v
                        results.tsv                progress.png
                        (experiment log)           (accuracy chart)
                                                        |
                                                        v
                        app.py <── best_model.pt ── checkpoint
                        (Gradio frontend)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams of the data pipeline, model architecture, evaluation pipeline, multi-step retrosynthesis inference, and AWS deployment.

### File structure

| File | Role | Who edits it |
|------|------|-------------|
| `prepare.py` | Data download (USPTO-50K), SMILES tokenizer, dataloader, evaluation function, building blocks extraction | Nobody (read-only) |
| `train.py` | GPT model, optimizer (AdamW), training loop, all hyperparameters, checkpoint saving | The AI agent |
| `program.md` | Agent instructions: experiment loop, keep/discard rules, metric, constraints | You (the researcher) |
| `app.py` | Gradio frontend: single-step and multi-step retrosynthesis with molecule visualization | You (for UI changes) |
| `analysis.ipynb` | Results visualization: progress chart, summary stats, top improvements | Run after experiments |
| `terraform/` | AWS infrastructure: g4dn.xlarge GPU instance (~$0.53/hr) | You (for deployment) |

### What's inside train.py

- **GPT model** -- decoder-only transformer with RoPE, RMS normalization, ReLU-squared activation, scaled dot-product attention (auto-dispatches to FlashAttention on supported hardware)
- **Sequence format** -- `<bos> [product SMILES] <sep> [reactant SMILES] <eos>`, with loss computed only on reactant tokens
- **Training loop** -- gradient accumulation, time-based progress tracking, LR scheduling (linear warmup + cosine warmdown), fast-fail on NaN/divergence
- **Outputs** -- model checkpoints (`best_model.pt`), per-step loss curve (`loss_curve.csv`), summary metrics

## Deployment Options

### AWS (recommended for autonomous agent runs)

Terraform config provisions a `g4dn.xlarge` (T4 16GB GPU, ~$0.53/hr):

```bash
cd terraform
terraform init
terraform apply -var="ssh_key_name=your-key"
# SSH in, start Claude Code in tmux, let it run overnight
```

| Duration | Cost | Experiments |
|----------|------|-------------|
| 1 hour | ~$0.53 | ~12 |
| 6 hours | ~$3.18 | ~72 |
| 12 hours | ~$6.36 | ~144 |

Stop instance (pause billing, keep data): `aws ec2 stop-instances --instance-ids <id>`
Start again: `aws ec2 start-instances --instance-ids <id>`
Destroy everything: `cd terraform && terraform destroy -var="ssh_key_name=your-key"`

### Google Colab (free, for testing and demos)

Open `colab.ipynb` in Colab, select T4 GPU runtime, run cells top to bottom. Free but limited: no autonomous agent (SSH blocked on free tier), sessions timeout after ~90 min idle.

### Local (MacBook, for development)

```bash
uv run prepare.py --tiny    # 500 reactions for fast testing
TIME_BUDGET=30 uv run train.py   # 30-second training run
uv run app.py               # Frontend at localhost:7860
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed comparison of all three deployment options.

## Key Design Decisions

These carry over from the original autoresearch framework and apply equally here:

- **Single file to modify.** The agent only touches `train.py`. Keeps scope manageable and diffs reviewable.
- **Fixed 5-minute time budget.** Makes every experiment directly comparable regardless of architecture/batch size changes. Also means autoresearch finds the best model *for your specific hardware*.
- **Self-contained.** One GPU, one file, one metric. No distributed training, no config files.

Additions specific to retrosynthesis:

- **SMILES canonicalization from day 0.** All molecules are canonicalized via RDKit during data prep, evaluation, and inference. Multi-reactant SMILES fragments are sorted alphabetically. No ambiguity.
- **Loss masking.** Only reactant tokens contribute to the loss -- the model attends to product tokens but isn't penalized for predicting them.
- **Bits-per-byte replaced by top-1 accuracy.** The metric is exact-match accuracy after SMILES canonicalization on 500 validation reactions.
- **Multi-step retrosynthesis at inference.** The model predicts one step; the frontend recursively applies it, stopping when reactants are commercially available (checked against building blocks extracted from training data).

## Results

After 12 autonomous experiments on a T4 GPU, the agent improved validation accuracy from 34.2% to 50.2%:

```
baseline:                34.2%
+ higher LR (1e-3):     45.0%  (+10.8%)
+ smaller batch (2^12): 45.8%  (+0.8%)
+ label smoothing:      50.2%  (+4.4%)
```

The 4 demo molecules (aspirin, caffeine, ibuprofen, adipic acid) are **not in the training data** -- any successful prediction is genuine generalization from learned reaction patterns.

See [HANDOFF.md](HANDOFF.md) for the full experiment log and next steps.

## Attribution

This project is an unofficial fork/adaptation of [autoresearch](https://github.com/karpathy/autoresearch) by [Andrej Karpathy](https://github.com/karpathy). The original project introduced the concept of autonomous AI-driven ML experimentation -- an agent that modifies training code, runs experiments on a fixed time budget, and keeps or discards changes based on a validation metric. The original trains a small GPT on web text (ClimbMix-400B) and measures validation bits-per-byte.

This adaptation preserves the core framework (the agent loop, keep/discard mechanism, fixed time budget, results logging, progress visualization) and applies it to a completely different domain -- chemical retrosynthesis prediction on USPTO-50K -- to demonstrate that the pattern generalizes beyond language model pretraining.

If you're interested in the original autoresearch for LLM pretraining, see:
- Repository: [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Context: [Karpathy's tweet on autoresearch](https://x.com/karpathy/status/2029701092347630069)
- Beginner guide: ["Dummy's Guide" to autoresearch](https://x.com/hooeem/status/2030720614752039185)

## License

MIT
