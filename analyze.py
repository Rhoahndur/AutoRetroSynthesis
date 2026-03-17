"""
Automated post-experiment analysis for the autoresearch agent loop.
Runs after every training experiment. Produces fixed-size outputs only.

Usage:
    uv run analyze.py --commit <hash> --accuracy <float> --validity <float> \
                      --memory <float> --status <keep|discard|invest|crash> \
                      --description "short description" \
                      [--config '{"DEPTH": 4, "LR": 0.001, ...}']

Outputs:
    analysis.txt       -- Fixed-size report (~50 lines) for the agent to read
    experiments.jsonl   -- Append-only full history (agent never reads directly)
    invest_state.json   -- Invest mechanism state (created/updated as needed)
    ideas.md            -- Truncated to 10 items if over limit
"""

import os
import json
import math
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_LOG = os.path.join(SCRIPT_DIR, "experiments.jsonl")
ANALYSIS_FILE = os.path.join(SCRIPT_DIR, "analysis.txt")
INVEST_STATE_FILE = os.path.join(SCRIPT_DIR, "invest_state.json")
IDEAS_FILE = os.path.join(SCRIPT_DIR, "ideas.md")
LOSS_CURVE_FILE = os.path.join(SCRIPT_DIR, "loss_curve.csv")

# ---------------------------------------------------------------------------
# Novelty score configuration
# ---------------------------------------------------------------------------

DIMENSIONS = {
    "DEPTH":          {"min": 1,    "max": 12,   "scale": "linear"},
    "N_EMBD":         {"min": 64,   "max": 512,  "scale": "linear"},
    "N_HEAD":         {"min": 1,    "max": 8,    "scale": "linear"},
    "LR":             {"min": 1e-5, "max": 1e-2, "scale": "log"},
    "BATCH_SIZE":     {"min": 2**8, "max": 2**18, "scale": "log"},
    "DROPOUT":        {"min": 0.0,  "max": 0.5,  "scale": "linear"},
    "WEIGHT_DECAY":   {"min": 0.0,  "max": 0.5,  "scale": "linear"},
    "LABEL_SMOOTH":   {"min": 0.0,  "max": 0.3,  "scale": "linear"},
    "WARMUP_RATIO":   {"min": 0.0,  "max": 0.2,  "scale": "linear"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_experiments():
    """Load all experiments from the JSONL log."""
    experiments = []
    if os.path.exists(EXPERIMENTS_LOG):
        with open(EXPERIMENTS_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    experiments.append(json.loads(line))
    return experiments


def load_invest_state():
    """Load invest mechanism state. Returns None if no active invest."""
    if not os.path.exists(INVEST_STATE_FILE):
        return None
    with open(INVEST_STATE_FILE) as f:
        state = json.load(f)
    if not state.get("active", False):
        return None
    return state


def save_invest_state(state):
    """Save invest state. Pass None or {"active": false} to clear."""
    if state is None:
        state = {"active": False}
    with open(INVEST_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def normalize_value(value, dim_config):
    """Normalize a hyperparameter value to [0, 1] given its dimension config."""
    lo, hi = dim_config["min"], dim_config["max"]
    if dim_config["scale"] == "log":
        if value <= 0 or lo <= 0 or hi <= 0:
            return 0.5
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        if hi == lo:
            return 0.5
        return (value - lo) / (hi - lo)


def config_to_vector(config):
    """Convert a config dict to a normalized vector for novelty computation."""
    vec = []
    for dim_name, dim_config in sorted(DIMENSIONS.items()):
        if dim_name in config:
            val = normalize_value(config[dim_name], dim_config)
            vec.append(max(0.0, min(1.0, val)))  # clamp to [0,1]
        else:
            vec.append(0.5)  # unknown = middle
    return vec


def euclidean_distance(v1, v2):
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def compute_novelty(config, all_experiments):
    """Compute novelty score: min distance to any previous experiment in normalized config space."""
    if not all_experiments:
        return 1.0
    current_vec = config_to_vector(config)
    min_dist = float("inf")
    for exp in all_experiments:
        exp_config = exp.get("config", {})
        if not exp_config:
            continue
        exp_vec = config_to_vector(exp_config)
        dist = euclidean_distance(current_vec, exp_vec)
        min_dist = min(min_dist, dist)
    return min_dist


def analyze_training_dynamics():
    """Analyze the latest loss curve for training dynamics."""
    if not os.path.exists(LOSS_CURVE_FILE):
        return {}

    steps, losses = [], []
    with open(LOSS_CURVE_FILE) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    steps.append(int(parts[0]))
                    losses.append(float(parts[1]))
                except ValueError:
                    continue

    if len(losses) < 10:
        return {"note": "too few steps for dynamics analysis"}

    first_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)

    # Trend: compare first half avg to second half avg
    mid = len(losses) // 2
    first_half_avg = sum(losses[:mid]) / mid
    second_half_avg = sum(losses[mid:]) / (len(losses) - mid)

    if second_half_avg < first_half_avg * 0.95:
        trend = "decreasing"
    elif second_half_avg > first_half_avg * 1.05:
        trend = "increasing (diverging)"
    else:
        trend = "flat (converged)"

    # End-of-run slope: linear regression on last 20% of steps
    tail_start = int(len(losses) * 0.8)
    tail = losses[tail_start:]
    if len(tail) >= 5:
        n = len(tail)
        x_mean = (n - 1) / 2
        y_mean = sum(tail) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(tail))
        den = sum((i - x_mean) ** 2 for i in range(n))
        end_slope = num / den if den > 0 else 0.0
    else:
        end_slope = 0.0

    return {
        "first_loss": round(first_loss, 4),
        "final_loss": round(final_loss, 4),
        "min_loss": round(min_loss, 4),
        "trend": trend,
        "end_slope": round(end_slope, 6),
        "total_steps": len(losses),
    }


def build_tried_configs_summary(experiments):
    """Build a one-line-per-dimension summary of all tried configurations."""
    # Collect values per dimension
    dim_values = {dim: [] for dim in DIMENSIONS}
    # Also track non-standard dimensions
    other_keys = set()

    for exp in experiments:
        config = exp.get("config", {})
        status = exp.get("status", "discard")
        for dim in DIMENSIONS:
            if dim in config:
                dim_values[dim].append((config[dim], status))
        for key in config:
            if key not in DIMENSIONS:
                other_keys.add(key)

    lines = []
    for dim in sorted(DIMENSIONS.keys()):
        values = dim_values[dim]
        if not values:
            lines.append(f"  {dim:16s} never tried")
            continue

        # Group by value, track best status for each
        val_status = {}
        for val, status in values:
            key = val
            if key not in val_status:
                val_status[key] = []
            val_status[key].append(status)

        parts = []
        for val in sorted(val_status.keys()):
            statuses = val_status[val]
            n_keep = statuses.count("keep")
            n_invest = statuses.count("invest")
            n_discard = statuses.count("discard")
            n_crash = statuses.count("crash")

            if isinstance(val, float):
                val_str = f"{val:.2e}" if val < 0.01 or val > 100 else f"{val:.4f}"
            else:
                val_str = str(val)

            if n_keep > 0:
                parts.append(f"{val_str} \u2713")
            elif n_invest > 0:
                parts.append(f"{val_str} ~")
            elif n_crash > 0:
                parts.append(f"{val_str} \u2717(crash)")
            else:
                mark = "\u2717" * min(n_discard, 3)
                parts.append(f"{val_str} {mark}")

        lines.append(f"  {dim:16s} {', '.join(parts)}")

    # Note dimensions never tried
    for key in sorted(other_keys):
        lines.append(f"  {key:16s} (non-standard dimension)")

    return lines


def detect_diminishing_returns(experiments):
    """Detect dimensions where recent experiments show diminishing returns."""
    if len(experiments) < 5:
        return []

    recent = experiments[-5:]
    dim_changes = {dim: 0 for dim in DIMENSIONS}
    dim_improvements = {dim: 0.0 for dim in DIMENSIONS}

    for i in range(1, len(recent)):
        prev_config = recent[i - 1].get("config", {})
        curr_config = recent[i].get("config", {})
        prev_acc = recent[i - 1].get("val_accuracy", 0)
        curr_acc = recent[i].get("val_accuracy", 0)
        delta = curr_acc - prev_acc

        for dim in DIMENSIONS:
            if prev_config.get(dim) != curr_config.get(dim):
                dim_changes[dim] += 1
                dim_improvements[dim] += delta

    warnings = []
    for dim in DIMENSIONS:
        if dim_changes[dim] >= 3:
            avg_improvement = dim_improvements[dim] / dim_changes[dim]
            if abs(avg_improvement) < 0.005:
                warnings.append(f"{dim}: last {dim_changes[dim]} changes averaged {avg_improvement:+.4f} (tapped out)")

    return warnings


def enforce_ideas_limit(max_items=10):
    """Truncate ideas.md to max_items if it exceeds the limit."""
    if not os.path.exists(IDEAS_FILE):
        return

    with open(IDEAS_FILE) as f:
        content = f.read()

    # Parse YAML-ish list: lines starting with "- "
    lines = content.strip().split("\n")
    header_lines = []
    item_lines = []
    in_items = False

    for line in lines:
        if line.strip().startswith("- "):
            in_items = True
        if in_items:
            item_lines.append(line)
        else:
            header_lines.append(line)

    # Count items (each "- " at indent level 0 starts a new item)
    items = []
    current_item = []
    for line in item_lines:
        if line.strip().startswith("- ") and current_item:
            items.append("\n".join(current_item))
            current_item = [line]
        else:
            current_item.append(line)
    if current_item:
        items.append("\n".join(current_item))

    if len(items) <= max_items:
        return  # no truncation needed

    # Keep first max_items (highest priority = first in list)
    truncated_items = items[:max_items]
    result = "\n".join(header_lines) + "\n" + "\n".join(truncated_items) + "\n"

    with open(IDEAS_FILE, "w") as f:
        f.write(result)

    print(f"analyze.py: truncated ideas.md from {len(items)} to {max_items} items")


# ---------------------------------------------------------------------------
# Main: log experiment and generate analysis
# ---------------------------------------------------------------------------


def log_experiment(commit, accuracy, validity, memory, status, description, config=None):
    """Append experiment to JSONL log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "val_accuracy": accuracy,
        "val_validity": validity,
        "memory_gb": memory,
        "status": status,
        "description": description,
        "config": config or {},
    }
    with open(EXPERIMENTS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def update_invest_state(status, accuracy):
    """Update invest mechanism based on experiment result."""
    state = load_invest_state()
    if state is None:
        if status == "invest":
            print("analyze.py: WARNING: invest status but no invest_state.json found")
        return

    # Decrement deadline
    state["deadline_remaining"] = state.get("deadline_remaining", 0) - 1

    # Check success: accuracy exceeded pre-invest best
    best_before = state.get("best_before_invest", 0)
    if accuracy > best_before:
        print(f"analyze.py: INVEST SUCCESS! accuracy {accuracy:.4f} > {best_before:.4f}")
        state["active"] = False
        state["outcome"] = "success"
        save_invest_state(state)
        return

    # Check abort: accuracy below threshold
    abort_threshold = state.get("abort_threshold", 0)
    if accuracy < abort_threshold:
        print(f"analyze.py: INVEST ABORT! accuracy {accuracy:.4f} < threshold {abort_threshold:.4f}")
        state["active"] = False
        state["outcome"] = "abort_threshold"
        save_invest_state(state)
        return

    # Check deadline
    if state["deadline_remaining"] <= 0:
        print(f"analyze.py: INVEST EXPIRED! deadline reached without exceeding {best_before:.4f}")
        state["active"] = False
        state["outcome"] = "deadline_expired"
        save_invest_state(state)
        return

    # Still active
    save_invest_state(state)
    print(f"analyze.py: invest active, {state['deadline_remaining']} experiments remaining")


def generate_analysis(current_experiment):
    """Generate the fixed-size analysis.txt report."""
    experiments = load_experiments()
    invest_state = load_invest_state()
    dynamics = analyze_training_dynamics()

    lines = []
    lines.append("=" * 60)
    lines.append("AUTORESEARCH ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total experiments: {len(experiments)}")
    lines.append("=" * 60)

    # Current run results
    lines.append("")
    lines.append("CURRENT RUN:")
    lines.append(f"  commit:       {current_experiment.get('commit', '?')}")
    lines.append(f"  val_accuracy: {current_experiment.get('val_accuracy', 0):.6f}")
    lines.append(f"  val_validity: {current_experiment.get('val_validity', 0):.6f}")
    lines.append(f"  memory_gb:    {current_experiment.get('memory_gb', 0):.1f}")
    lines.append(f"  status:       {current_experiment.get('status', '?')}")
    lines.append(f"  description:  {current_experiment.get('description', '?')}")

    # Training dynamics
    if dynamics:
        lines.append("")
        lines.append("TRAINING DYNAMICS:")
        for key, val in dynamics.items():
            lines.append(f"  {key}: {val}")

    # Best experiment
    if experiments:
        best = max(experiments, key=lambda e: e.get("val_accuracy", 0))
        lines.append("")
        lines.append("BEST EXPERIMENT:")
        lines.append(f"  commit:       {best.get('commit', '?')}")
        lines.append(f"  val_accuracy: {best.get('val_accuracy', 0):.6f}")
        lines.append(f"  description:  {best.get('description', '?')}")

        # Rank
        accuracies = sorted([e.get("val_accuracy", 0) for e in experiments], reverse=True)
        current_acc = current_experiment.get("val_accuracy", 0)
        rank = 1
        for a in accuracies:
            if a > current_acc:
                rank += 1
            else:
                break
        lines.append(f"  current rank: {rank}/{len(experiments)}")

    # Last 10 experiments
    lines.append("")
    lines.append("RECENT EXPERIMENTS (last 10):")
    lines.append(f"  {'commit':8s} {'accuracy':10s} {'validity':10s} {'status':8s} description")
    for exp in experiments[-10:]:
        lines.append(
            f"  {exp.get('commit', '?'):8s} "
            f"{exp.get('val_accuracy', 0):.6f}   "
            f"{exp.get('val_validity', 0):.6f}   "
            f"{exp.get('status', '?'):8s} "
            f"{exp.get('description', '?')}"
        )

    # Tried configurations
    lines.append("")
    lines.append("TRIED CONFIGURATIONS:")
    lines.extend(build_tried_configs_summary(experiments))

    # Novelty score
    if current_experiment.get("config"):
        prev_experiments = experiments[:-1] if len(experiments) > 1 else []
        novelty = compute_novelty(current_experiment["config"], prev_experiments)
        lines.append("")
        lines.append(f"NOVELTY SCORE: {novelty:.3f}")
        if novelty < 0.1:
            lines.append("  LOW -- very similar to a previous experiment, try a different direction")
        elif novelty < 0.3:
            lines.append("  MODERATE -- some variation from previous experiments")
        else:
            lines.append("  HIGH -- exploring new territory")

    # Diminishing returns
    dr_warnings = detect_diminishing_returns(experiments)
    if dr_warnings:
        lines.append("")
        lines.append("DIMINISHING RETURNS:")
        for w in dr_warnings:
            lines.append(f"  {w}")

    # Invest state
    if invest_state:
        lines.append("")
        lines.append("INVEST STATE:")
        lines.append(f"  reason:    {invest_state.get('reason', '?')}")
        lines.append(f"  revert_to: {invest_state.get('revert_commit', '?')}")
        lines.append(f"  remaining: {invest_state.get('deadline_remaining', 0)} experiments")
        lines.append(f"  threshold: {invest_state.get('abort_threshold', 0):.4f}")
        lines.append(f"  target:    exceed {invest_state.get('best_before_invest', 0):.4f}")

    lines.append("")
    lines.append("=" * 60)

    # Write analysis
    with open(ANALYSIS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"analyze.py: wrote {ANALYSIS_FILE} ({len(lines)} lines)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-experiment analysis for autoresearch")
    parser.add_argument("--commit", required=True, help="Git commit hash (short)")
    parser.add_argument("--accuracy", type=float, required=True, help="val_accuracy")
    parser.add_argument("--validity", type=float, required=True, help="val_validity")
    parser.add_argument("--memory", type=float, required=True, help="Peak memory in GB")
    parser.add_argument("--status", required=True, choices=["keep", "discard", "invest", "crash"])
    parser.add_argument("--description", required=True, help="Short experiment description")
    parser.add_argument("--config", default="{}", help="JSON string of hyperparameter config")

    args = parser.parse_args()

    config = json.loads(args.config)

    # 1. Log experiment
    entry = log_experiment(
        commit=args.commit,
        accuracy=args.accuracy,
        validity=args.validity,
        memory=args.memory,
        status=args.status,
        description=args.description,
        config=config,
    )
    print(f"analyze.py: logged experiment {args.commit} ({args.status})")

    # 2. Update invest state if active
    update_invest_state(args.status, args.accuracy)

    # 3. Enforce ideas.md size limit
    enforce_ideas_limit(max_items=10)

    # 4. Generate analysis report
    generate_analysis(entry)
