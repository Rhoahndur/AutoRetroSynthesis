# autoresearch — retrosynthesis

This is an experiment to have an LLM autonomously improve a retrosynthesis prediction model.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, SMILES tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch-retro/data/` contains `train_data.pt`, `val_data.pt`, and `vocab.json`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task

The model is a small GPT (decoder-only transformer) that learns **retrosynthesis**: given a target molecule (SMILES string), predict the reactants needed to synthesize it.

Training data is USPTO-50K (50K reactions from US patents). The model sees sequences formatted as:
```
<bos> [product SMILES tokens] <sep> [reactant SMILES tokens] <eos>
```

Loss is computed only on the reactant tokens (product tokens are masked). At evaluation time, the model is given a product and autoregressively generates the predicted reactants.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_retro_accuracy` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_accuracy.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful accuracy gains.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as is.

**Experiment ideas** (try these and more):
- Model size: sweep DEPTH (2, 4, 6, 8), N_EMBD (128, 256, 384, 512)
- Learning rate: sweep LEARNING_RATE (1e-4, 3e-4, 1e-3, 3e-3)
- Batch size: try TOTAL_BATCH_SIZE from 2**12 to 2**16
- Warmup/warmdown ratios
- Optimizer: try different betas, weight decay values
- Architecture: add dropout, try different activation functions (GELU, SwiGLU)
- Multi-query attention (fewer KV heads)
- Data augmentation: SMILES strings have multiple valid representations for the same molecule. During training, randomly generate non-canonical SMILES using RDKit to augment the data (note: RDKit is available as it's in the dependencies).
- Reaction class conditioning: the dataset has reaction class labels — try prepending a class token.
- Label smoothing in the loss function
- Gradient clipping

## Output format

Once the script finishes it prints a summary like this:

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

Extract the key metric:
```
grep "^val_accuracy:" run.log
```

## Logging results

Log each experiment to `results.tsv` (tab-separated, NOT comma-separated).

Header row and 6 columns:

```
commit	val_accuracy	val_validity	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_accuracy achieved (e.g. 0.123456) — use 0.000000 for crashes
3. val_validity (fraction of valid SMILES generated) — use 0.000000 for crashes
4. peak memory in GB, round to .1f — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_accuracy	val_validity	memory_gb	status	description
a1b2c3d	0.120000	0.850000	2.1	keep	baseline
b2c3d4e	0.185000	0.890000	2.1	keep	increase LR to 1e-3
c3d4e5f	0.105000	0.820000	2.1	discard	switch to GeLU activation
d4e5f6g	0.000000	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^val_accuracy:\|^val_validity:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to diagnose.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file)
8. If val_accuracy improved (higher), you "advance" the branch, keeping the git commit
9. If val_accuracy is equal or worse, you git reset back to where you started

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: Fix typos/imports and re-run. If the idea is fundamentally broken, skip it.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
