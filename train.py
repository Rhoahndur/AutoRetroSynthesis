"""
Retrosynthesis autoresearch training script. Single-GPU, single-file.
Adapted from nanochat/autoresearch for retrosynthesis prediction.
Usage: uv run train.py
"""

import os
import csv
import gc
import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET as _TIME_BUDGET, SMILESTokenizer,
    make_dataloader, evaluate_retro_accuracy,
)
from model import GPT, GPTConfig

# Allow overriding time budget via environment variable (for local testing)
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", _TIME_BUDGET))

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 4               # number of transformer layers
N_EMBD = 256            # model embedding dimension
N_HEAD = 4              # number of attention heads
HEAD_DIM = N_EMBD // N_HEAD  # = 64

# Optimization
TOTAL_BATCH_SIZE = 2**14  # ~16K tokens per optimizer step
LEARNING_RATE = 3e-4      # peak learning rate (AdamW)
WEIGHT_DECAY = 0.1        # AdamW weight decay
ADAM_BETAS = (0.9, 0.95)  # Adam betas
WARMUP_RATIO = 0.05       # fraction of time for LR warmup
WARMDOWN_RATIO = 0.3      # fraction of time for LR cosine warmdown
FINAL_LR_FRAC = 0.1       # final LR as fraction of peak

# Batch
DEVICE_BATCH_SIZE = 64    # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    print(f"Device: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: Apple MPS")
else:
    device = torch.device("cpu")
    print("Device: CPU")

use_cuda = device.type == "cuda"
use_compile = use_cuda  # only compile on CUDA
use_autocast = use_cuda or device.type == "cpu"

# T4 and older GPUs don't support bfloat16 -- use float16 instead
if use_cuda:
    capability = torch.cuda.get_device_capability()
    autocast_dtype = torch.bfloat16 if capability >= (8, 0) else torch.float16
else:
    autocast_dtype = torch.bfloat16

if use_autocast:
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=autocast_dtype)
else:
    from contextlib import nullcontext
    autocast_ctx = nullcontext()

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

tokenizer = SMILESTokenizer.from_file()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_embd=N_EMBD,
)
print(f"Model config: {asdict(config)}")

model = GPT(config).to(device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

# Keep a reference to the raw model for generation (torch.compile wraps forward)
raw_model = model

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * (MAX_SEQ_LEN - 1)
grad_accum_steps = max(1, TOTAL_BATCH_SIZE // tokens_per_fwdbwd)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=ADAM_BETAS,
    weight_decay=WEIGHT_DECAY,
)

if use_compile:
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(DEVICE_BATCH_SIZE, "train", device)
x, y, mask, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation: {grad_accum_steps}")
print(f"Effective batch size: {grad_accum_steps * tokens_per_fwdbwd} tokens")

# LR schedule
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Safety guardrails (hard limits the agent cannot override)
# ---------------------------------------------------------------------------
MAX_WALL_CLOCK = TIME_BUDGET * 2 + 120  # absolute wall-clock kill (2x budget + 2min startup)
MAX_PARAMS = 200_000_000                 # 200M params max (prevent OOM)
if num_params > MAX_PARAMS:
    print(f"FAIL: model has {num_params/1e6:.1f}M params, exceeds {MAX_PARAMS/1e6:.0f}M limit")
    exit(1)

loss_log = []  # (step, loss) pairs for loss curve CSV
t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
warmup_steps = 5  # skip first N steps from time budget (compilation overhead)

while True:
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y, loss_mask=mask)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, mask, epoch = next(train_loader)

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0.0
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE * lrm

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: loss diverged
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("\nFAIL: loss diverged")
        exit(1)

    # Hard wall-clock kill (agent cannot override this)
    if time.time() - t_start > MAX_WALL_CLOCK:
        print(f"\nFAIL: exceeded {MAX_WALL_CLOCK}s wall-clock limit")
        exit(1)

    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step >= warmup_steps:
        total_training_time += dt

    # Loss curve logging
    loss_log.append((step, train_loss_f))

    # Smoothed loss for display
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0

    print(f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | lr: {LEARNING_RATE * lrm:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        if use_cuda:
            gc.freeze()
            gc.disable()

    step += 1

    if step >= warmup_steps and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r log

total_tokens = step * TOTAL_BATCH_SIZE

# ---------------------------------------------------------------------------
# Save loss curve
# ---------------------------------------------------------------------------

with open("loss_curve.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss"])
    writer.writerows(loss_log)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

raw_model.eval()
print("Evaluating retrosynthesis accuracy on validation set...")
val_accuracy, val_validity = evaluate_retro_accuracy(raw_model, device)
print(f"val_accuracy={val_accuracy:.6f}, val_validity={val_validity:.6f}")

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------

checkpoint = {
    "model_state_dict": raw_model.state_dict(),
    "config": asdict(config),
    "val_accuracy": val_accuracy,
    "val_validity": val_validity,
}
torch.save(checkpoint, "latest_model.pt")

# Save as best if improved
best_path = "best_model.pt"
save_best = True
if os.path.exists(best_path):
    try:
        prev = torch.load(best_path, map_location="cpu", weights_only=True)
        if prev.get("val_accuracy", 0) >= val_accuracy:
            save_best = False
    except Exception:
        pass
if save_best:
    torch.save(checkpoint, best_path)
    print(f"New best model saved (accuracy={val_accuracy:.6f})")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if use_cuda else 0

print("---")
print(f"val_accuracy:     {val_accuracy:.6f}")
print(f"val_validity:     {val_validity:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
