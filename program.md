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
   - `analyze.py` — post-experiment analysis. You run this after each experiment. Do not modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch-retro/data/` contains `train_data.pt`, `val_data.pt`, and `vocab.json`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize files**: Create `ideas.md` with 5-10 initial experiment ideas as a YAML list. Create empty `invest_state.json` with `{"active": false}`.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task

The model is a small GPT (decoder-only transformer) that learns **retrosynthesis**: given a target molecule (SMILES string), predict the reactants needed to synthesize it.

Training data is USPTO-50K (50K reactions from US patents). The model sees sequences formatted as:
```
<bos> [product SMILES tokens] <sep> [reactant SMILES tokens] <eos>
```

If reaction class conditioning is enabled, format is:
```
<bos> <class_N> [product SMILES tokens] <sep> [reactant SMILES tokens] <eos>
```

Loss is computed only on the reactant tokens (product and class tokens are masked). At evaluation time, the model is given a product and autoregressively generates the predicted reactants.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants.
- Modify `analyze.py`. It is read-only. It produces the analysis reports you consume.
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

Extract the key metrics:
```
grep "^val_accuracy:\|^val_validity:\|^peak_vram_mb:" run.log
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15`).

LOOP FOREVER:

### 1. Read context
Read `analysis.txt` (if it exists from a prior experiment). This is a fixed-size report (~50 lines) containing:
- Last experiment results and training dynamics
- Best experiment so far
- Last 10 experiments table
- Tried configurations summary (what's been tried for each hyperparameter)
- Novelty score (how different the last experiment was from all previous)
- Diminishing returns warnings
- Invest mechanism state (if active)

Also read `ideas.md` for your prioritized experiment backlog.

### 2. Check invest state
If `invest_state.json` shows an active invest:
- Continue pursuing the invest strategy (the invest has a reason and deadline)
- `analyze.py` will automatically check abort/success conditions after each run
- If invest was just resolved (success or failure), adjust your strategy accordingly

### 3. Formulate hypothesis
Based on the analysis, formulate your next experiment. Consider:
- **Tried configs**: Don't retry configurations that already failed (check the summary)
- **Novelty score**: If recent experiments have low novelty, try a fundamentally different direction
- **Diminishing returns**: If a dimension is flagged as tapped out, move to a different one
- **Ideas list**: Check your prioritized ideas for the next thing to try
- **Training dynamics**: If loss was still decreasing at end of run, the model may benefit from more capacity or faster training

### 4. Execute experiment
1. Edit `train.py` with your experimental change (one focused change per experiment)
2. `git add train.py && git commit -m "descriptive message"`
3. `uv run train.py > run.log 2>&1`
4. Read results: `grep "^val_accuracy:\|^val_validity:\|^peak_vram_mb:" run.log`
5. If grep is empty, the run crashed. Run `tail -n 50 run.log` to diagnose.

### 5. Analyze and decide
Extract the commit hash and results, then run analyze.py:

```bash
COMMIT=$(git rev-parse --short HEAD)
# Extract values from run.log
ACC=$(grep "^val_accuracy:" run.log | awk '{print $2}')
VAL=$(grep "^val_validity:" run.log | awk '{print $2}')
MEM=$(grep "^peak_vram_mb:" run.log | awk '{printf "%.1f", $2/1024}')

# Build config JSON from train.py (extract current hyperparameters)
CONFIG=$(python3 -c "
import re
with open('train.py') as f: src = f.read()
config = {}
for var in ['DEPTH', 'N_EMBD', 'N_HEAD', 'DROPOUT', 'TOTAL_BATCH_SIZE', 'LEARNING_RATE', 'WEIGHT_DECAY', 'WARMUP_RATIO', 'WARMDOWN_RATIO', 'FINAL_LR_FRAC']:
    m = re.search(rf'^{var}\s*=\s*(.+?)(?:\s*#|$)', src, re.M)
    if m:
        try: config[var] = eval(m.group(1).strip())
        except: pass
import json; print(json.dumps(config))
")
```

**Decide status:**
- **keep**: val_accuracy improved over the previous best
- **discard**: val_accuracy is equal or worse, and this isn't a foundational change
- **invest**: val_accuracy dropped, BUT you believe this change is foundational (e.g., SMILES augmentation, architecture change that needs follow-up tuning). You must create `invest_state.json`:
  ```json
  {
    "active": true,
    "reason": "why this investment will pay off",
    "revert_commit": "<commit hash to revert to if invest fails>",
    "best_before_invest": <accuracy before invest>,
    "deadline_remaining": 3,
    "abort_threshold": <70% of best accuracy>
  }
  ```
  **Max 1 active invest at a time. Default deadline: 3 experiments.**
- **crash**: the run failed

Run analyze.py:
```bash
uv run analyze.py --commit "$COMMIT" --accuracy "$ACC" --validity "$VAL" \
  --memory "$MEM" --status "keep" --description "short description" \
  --config "$CONFIG"
```

**If discard**: `git reset --hard HEAD~1`
**If invest**: keep the commit, create invest_state.json
**If keep**: keep the commit, save as best if applicable

### 6. Update ideas
Update `ideas.md`: remove the idea you just tried, add any new ideas that occurred to you based on the results. Keep the list prioritized (most promising first). Maximum 10 items — if you have more, drop the lowest priority ones.

### 7. Repeat
Go back to step 1. **NEVER STOP.** The loop runs until the human interrupts you.

## Investigation mode (when stuck)

If you've had 3+ consecutive experiments with no improvement:

1. Write a short Python script to analyze your experiment history more deeply (e.g., plot loss curves, compare training dynamics across runs, look for patterns in what works)
2. Save any plots to the project directory
3. Use the insights to formulate a more informed hypothesis
4. Consider using the **invest** mechanism to try a foundational change that might not pay off immediately

## Literature search (when plateaued)

If you've had 5+ consecutive experiments with no improvement AND investigation mode hasn't helped:

1. Search for recent retrosynthesis ML papers (arXiv, Semantic Scholar) for techniques you haven't tried
2. Focus on: SMILES-based models, transformer retrosynthesis, data augmentation techniques, training tricks
3. Filter each idea through a feasibility check: Can I implement this by only editing `train.py`? Will it work within a 5-minute training budget?
4. Add promising ideas to `ideas.md` with a brief note on the source

This is a last resort — most of the time, investigation mode and the analysis report will provide enough guidance.

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: Fix typos/imports and re-run. If the idea is fundamentally broken, skip it.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes, enter investigation mode, or search the literature. The loop runs until the human interrupts you, period.
