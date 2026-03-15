# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

---

## What is autoresearch?

Autoresearch flips the traditional ML research workflow: instead of a human manually tweaking training code, you write a research strategy in Markdown (`program.md`) and let an AI agent experiment autonomously. The agent modifies the training script, runs a 5-minute experiment, checks if validation loss improved, keeps or discards the change, and loops forever. Go to sleep, wake up to ~100 completed experiments and a better model.

The training code is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). More context in this [tweet](https://x.com/karpathy/status/2029701092347630069).

---

## Quick Start

### Requirements

- Single NVIDIA GPU (tested on H100; see [Platform Support](#platform-support) for smaller hardware)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Install and prepare

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py
```

### 2. Run a single experiment manually

```bash
uv run train.py
```

This trains a small GPT model for 5 minutes and prints a summary:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

The primary metric is **val_bpb** (validation bits per byte) -- lower is better.

### 3. Go autonomous

Point your AI coding agent (Claude Code, Codex, etc.) at the repo with permissions disabled, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The agent will create a branch, establish a baseline, then loop indefinitely -- modifying `train.py`, running experiments, keeping improvements, discarding regressions, and logging everything to `results.tsv`.

---

## Use Cases

### Overnight autonomous research

The primary use case. Leave the agent running while you sleep. Each experiment takes ~5 minutes, so you get ~12 experiments/hour or ~100 overnight. The agent explores architecture changes, hyperparameter sweeps, optimizer tweaks, and activation functions -- all without human intervention.

### Interactive experimentation

Use autoresearch as a rapid prototyping bench. Tell the agent a specific hypothesis ("try SwiGLU instead of ReLU-squared", "double the depth and halve the width") and watch it test the idea in 5 minutes. The fixed time budget makes every experiment directly comparable regardless of what changed.

### Learning and teaching

The entire training pipeline fits in two files. `prepare.py` covers tokenization, data loading, and evaluation. `train.py` covers the full transformer architecture, a modern optimizer (Muon + AdamW), and the training loop. Reading and modifying these files is a hands-on crash course in LLM pretraining. If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) provides a lot more context.

### Benchmarking your GPU

Since training runs for a fixed wall-clock budget, the resulting `val_bpb` reflects how much useful work your specific GPU can do in 5 minutes. Run the baseline on different hardware to compare effective throughput in a realistic training workload.

---

## Architecture

```
 You (the human)
  |
  |  write research strategy
  v
 program.md ──────────────> AI Agent (Claude, Codex, etc.)
                               |
                               |  modify code, commit, run, evaluate
                               v
                            train.py ──> 5-min training run ──> val_bpb
                               |                                  |
                               |  if improved: keep commit        |
                               |  if worse: git reset             |
                               v                                  v
                            results.tsv                    analysis.ipynb
                            (experiment log)               (visualization)
```

### The three files that matter

| File | Role | Who edits it |
|------|------|-------------|
| `prepare.py` | Fixed constants, data download, BPE tokenizer training, dataloader, evaluation (`evaluate_bpb`) | Nobody (read-only) |
| `train.py` | GPT model, Muon+AdamW optimizer, training loop, all hyperparameters | The AI agent |
| `program.md` | Research strategy, constraints, experiment loop protocol | You (the human) |

### What's inside train.py

- **GPT model** -- transformer with RoPE, Flash Attention 3, multi-query attention, value embeddings (ResFormer-style), ReLU-squared activation, alternating sliding/full attention windows
- **MuonAdamW optimizer** -- Muon (orthogonalization via Polar Express) for weight matrices, AdamW for everything else, with Nesterov momentum and cautious weight decay
- **Training loop** -- gradient accumulation, time-based progress tracking, LR scheduling (linear warmup + cosine warmdown), fast-fail on NaN/divergence, `torch.compile`

### Key design decisions

- **Single file to modify.** Keeps scope manageable and diffs reviewable.
- **Fixed 5-minute time budget.** Makes every experiment directly comparable regardless of architecture/batch size changes. Also means autoresearch finds the best model *for your specific hardware*.
- **Bits-per-byte metric.** Vocab-size-independent, so changing the tokenizer or vocabulary doesn't break comparisons.
- **Self-contained.** One GPU, one file, one metric. No distributed training, no config files.

---

## Project Ideas

Here are some fun directions to take autoresearch:

### Multi-agent research swarm
Run multiple agents on separate GPUs, each exploring a different research direction (one focused on architecture, another on optimization, a third on regularization). Have a coordinator agent merge the best findings across branches. You're now running your own AI research lab.

### Competitive leaderboard
Set up a shared `results.tsv` across a group of friends or a class. Everyone runs autoresearch on their own hardware for 8 hours. Compare final `val_bpb` scores. Who wrote the best `program.md`? Whose agent found the most creative optimization?

### Research strategy meta-optimization
The `program.md` is itself code -- just for agents instead of CPUs. Write a meta-agent that *modifies program.md*, measures how fast the inner agent improves `val_bpb` over N experiments, and iterates. Evolve the research strategy itself.

### Scaling law explorer
Modify the time budget and model size systematically. Sweep across DEPTH values (2, 4, 8, 16) and time budgets (1 min, 5 min, 15 min). Plot the Pareto frontier of compute vs. loss. Let the agent discover chinchilla-optimal configurations for your GPU.

### Architecture search from scratch
Strip `train.py` down to a bare skeleton (embedding + linear output) and see what the agent rebuilds from nothing. Does it rediscover attention? Multi-head attention? Layer normalization? RoPE? Document the evolutionary path.

### Curriculum learning experiments
Fork `prepare.py` to support multiple datasets (TinyStories, code, Wikipedia, math). Let the agent control dataset mixing ratios in `train.py` and discover optimal data curricula within the 5-minute window.

---

## Platform Support

This code requires a single NVIDIA GPU. For smaller hardware (MacBooks, consumer GPUs), see the forks below and consider these tuning recommendations:

1. Use a lower-entropy dataset like [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) for reasonable results with small models
2. Decrease `vocab_size` (down to 4096, 2048, or even 256 for byte-level)
3. Lower `MAX_SEQ_LEN` in `prepare.py` (down to 256) and increase `DEVICE_BATCH_SIZE` in `train.py` to compensate
4. Decrease `EVAL_TOKENS` in `prepare.py` for faster validation
5. Lower `DEPTH` (e.g. from 8 to 4) -- this is the primary complexity knob
6. Use `WINDOW_PATTERN = "L"` (full attention) instead of `"SSSL"` (alternating banded)
7. Lower `TOTAL_BATCH_SIZE` (e.g. to `2**14`)

### Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (macOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (macOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

---

## Analyzing Results

After a run, open `analysis.ipynb` to visualize experiment progress:

```bash
uv run jupyter lab analysis.ipynb
```

The notebook loads `results.tsv`, plots `val_bpb` over time with the running-best frontier highlighted, annotates kept experiments, and ranks improvements by delta. Useful for understanding which changes had the biggest impact.

You can also inspect results from the command line:

```bash
# View all results
cat results.tsv

# Check latest run
grep "^val_bpb:" run.log

# Debug a crash
tail -n 50 run.log
```

---

## Project Structure

```
prepare.py      -- constants, data prep + runtime utilities (do not modify)
train.py        -- model, optimizer, training loop (agent modifies this)
program.md      -- agent instructions (human modifies this)
analysis.ipynb  -- results visualization
results.tsv     -- experiment log (generated, not committed)
pyproject.toml  -- dependencies
```

## License

MIT
