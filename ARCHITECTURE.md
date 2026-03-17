# RetroSynth Autoresearch -- Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEVELOPER MACHINE (MacBook Air M3)          │
│                                                                 │
│  Code Development          Terraform          Frontend          │
│  ┌──────────┐             ┌──────────┐       ┌──────────┐      │
│  │prepare.py│             │ main.tf  │       │  app.py  │      │
│  │ train.py │             │          │       │ (Gradio) │      │
│  │analyze.py│             │ terraform│       │ + Plotly  │      │
│  │program.md│             │  apply   │       │ + beam   │      │
│  └────┬─────┘             └────┬─────┘       └────┬─────┘      │
│       │ local test              │ provisions       │ loads      │
│       │ (CPU, tiny data)        │                  │ best_model │
│       v                         │                  │ .pt        │
│  Verify pipeline                │                  v            │
│  works end-to-end               │           Molecule input       │
│                                 │           → retrosynthesis     │
│                                 │             route display      │
└────────────────────────────────┼────────────────────────────────┘
                                 │
                    SSH / SCP    │
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                    AWS g4dn.xlarge (T4 16GB)                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              AUTORESEARCH AGENT LOOP (enhanced)          │    │
│  │                                                         │    │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐            │    │
│  │  │  Read    │──>│Formulate │──>│  Modify  │            │    │
│  │  │analysis  │   │hypothesis│   │ train.py │            │    │
│  │  │.txt      │   │(novelty- │   └────┬─────┘            │    │
│  │  │+ ideas   │   │ guided)  │        │                  │    │
│  │  └──────────┘   └──────────┘        v                  │    │
│  │       ^                        ┌──────────┐            │    │
│  │       │                        │  Train   │            │    │
│  │       │                        │  5 min   │            │    │
│  │       │                        └────┬─────┘            │    │
│  │       │                             │                  │    │
│  │       │                        ┌────▼─────┐            │    │
│  │       │                        │ Evaluate │            │    │
│  │       │                        │ accuracy │            │    │
│  │       │                        └────┬─────┘            │    │
│  │       │                             │                  │    │
│  │       │                        ┌────▼─────┐            │    │
│  │       │                        │analyze.py│            │    │
│  │       │                        │ - log    │            │    │
│  │       │                        │ - invest │            │    │
│  │       │                        │ - report │            │    │
│  │       │                        └────┬─────┘            │    │
│  │       │                             │                  │    │
│  │       │          ┌──────────────────┤                  │    │
│  │       │          │          │       │                  │    │
│  │       │     improved?   foundational?  not improved?   │    │
│  │       │          │          │       │                  │    │
│  │       │          v          v       v                  │    │
│  │       │    ┌────────┐ ┌────────┐ ┌────────┐           │    │
│  │       │    │  KEEP  │ │ INVEST │ │DISCARD │           │    │
│  │       │    │ commit │ │deadline│ │git rset│           │    │
│  │       │    └────────┘ └────────┘ └────────┘           │    │
│  │       │          │          │       │                  │    │
│  │       │          v          v       v                  │    │
│  │       │    ┌──────────────────────────────┐            │    │
│  │       └────┤  Update ideas.md, repeat     │            │    │
│  │            └──────────────────────────────┘            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Outputs:                                                       │
│  ├── experiments.jsonl    (full history, append-only)            │
│  ├── analysis.txt         (fixed-size report for agent)         │
│  ├── invest_state.json    (invest mechanism state)              │
│  ├── ideas.md             (max 10 items, mechanically capped)   │
│  ├── best_model.pt        (best checkpoint)                     │
│  ├── loss_curve.csv       (per-experiment training curves)      │
│  └── results.tsv          (legacy experiment log)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Pipeline

```
USPTO-50K (HuggingFace)
       │
       v
┌──────────────────────────────────────────────────────────┐
│                    prepare.py                             │
│                                                          │
│  1. Download USPTO-50K dataset                           │
│     └── 50K reactions: product >> reactant SMILES        │
│                                                          │
│  2. Canonicalize all SMILES (RDKit)                      │
│     └── Sort multi-reactant fragments alphabetically     │
│                                                          │
│  3. Extract reaction class labels (--reaction-class)     │
│     └── 10 classes: alkylation, acylation, C-C bond, ..  │
│                                                          │
│  4. Build SMILES tokenizer (character-level, regex)      │
│     └── ~90 SMILES token vocab                           │
│     └── Special tokens: <pad>=0, <bos>=1, <eos>=2,      │
│         <sep>=3, <class_0>=4, ..., <class_9>=13          │
│                                                          │
│  5. SMILES Augmentation (--augment N, training only)     │
│     └── Generate N random SMILES per reaction via RDKit  │
│     └── Randomize fragment order for reactants           │
│     └── Multiplies training data by (1+N)x               │
│                                                          │
│  6. Tokenize all reactions                               │
│     └── <bos> [<class_N>] [product] <sep> [react] <eos>  │
│                                                          │
│  7. Save as tensors                                      │
│     └── train.pt, val.pt, test.pt                        │
│     └── vocab.json, raw_reactions.json                   │
│                                                          │
│  8. Extract building blocks                              │
│     └── Reactants appearing in >= 3 reactions            │
│     └── + common lab reagents                            │
│     └── building_blocks.pkl                              │
└──────────────────────────────────────────────────────────┘
```

## Model Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    GPT (Decoder-Only)                     │
│                                                          │
│  Input: <bos> <class_2> C C ( = O ) O c 1 ... <sep>     │
│                                                          │
│  ┌──────────────────────────────────┐                    │
│  │  Token Embedding (vocab=~90+14) │                    │
│  │  + Rotary Position Embedding     │                    │
│  └──────────────┬───────────────────┘                    │
│                 │                                        │
│  ┌──────────────▼───────────────────┐                    │
│  │  Transformer Block x4            │                    │
│  │  ┌─────────────────────────┐     │                    │
│  │  │ RMSNorm                 │     │  n_embd = 256      │
│  │  │ Causal Self-Attention   │     │  head_dim = 64     │
│  │  │ (SDPA, n_head=4)       │     │  ~2-5M params      │
│  │  │ + Dropout               │     │                    │
│  │  ├─────────────────────────┤     │                    │
│  │  │ RMSNorm                 │     │                    │
│  │  │ MLP (ReLU²)            │     │                    │
│  │  │ + Dropout               │     │                    │
│  │  └─────────────────────────┘     │                    │
│  └──────────────┬───────────────────┘                    │
│                 │                                        │
│  ┌──────────────▼───────────────────┐                    │
│  │  RMSNorm → Linear → Softmax     │                    │
│  └──────────────┬───────────────────┘                    │
│                 │                                        │
│  Output: O c 1 c c c c c 1 C ( = O ) O <eos>            │
│                                                          │
│  Loss: cross-entropy on reactant tokens only             │
│  (product, class, and <pad> tokens masked)               │
└──────────────────────────────────────────────────────────┘
```

## Training Sequence Format

```
Without reaction class:
Position:  0    1  2  3  4  5  6  7  8  9  10  11 12 13 14 ...
Token:    <bos> C  C  (  =  O  )  O  c  1  c   c  c  c  <sep> O ...
Loss mask: 0    0  0  0  0  0  0  0  0  0  0   0  0  0   0    1 ...
           |<-- product (no loss) -->|                    |<-- reactants (loss) -->|

With reaction class:
Position:  0     1        2  3  4  5  6  7  8  ...  <sep> O ...
Token:    <bos> <class_2> C  C  (  =  O  )  O  ...  <sep> O ...
Loss mask: 0     0        0  0  0  0  0  0  0  ...   0    1 ...
           |<-- class + product (no loss) -->|       |<-- reactants (loss) -->|
```

## Evaluation Pipeline

```
┌──────────────────────────────────────────────────────────┐
│              evaluate_retro_accuracy()                    │
│                                                          │
│  For 500 val examples:                                   │
│                                                          │
│  1. Feed prefix: <bos> [class] [product tokens] <sep>    │
│                                                          │
│  2. Generation (configurable):                           │
│     ├── Greedy: argmax next token until <eos>            │
│     └── Beam search (width=10): explore top-K candidates │
│                                                          │
│  3. Decode generated tokens → SMILES string              │
│                                                          │
│  4. Canonicalize (RDKit):                                │
│     ├── Split on "."                                     │
│     ├── Canonicalize each fragment                       │
│     ├── Sort alphabetically                              │
│     └── Rejoin with "."                                  │
│                                                          │
│  5. Compare to ground truth (also canonicalized)         │
│     └── Exact string match                               │
│                                                          │
│  Metrics (greedy mode):                                  │
│  ├── val_accuracy = correct / total (top-1)              │
│  └── val_validity = valid_smiles / total                 │
│                                                          │
│  Metrics (beam mode):                                    │
│  ├── top-1 accuracy                                      │
│  ├── top-3 accuracy                                      │
│  ├── top-5 accuracy                                      │
│  ├── top-10 accuracy                                     │
│  └── val_validity                                        │
└──────────────────────────────────────────────────────────┘
```

## Post-Experiment Analysis Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                      analyze.py                           │
│                                                          │
│  Input: --commit, --accuracy, --status, --config (JSON)  │
│                                                          │
│  1. Log experiment to experiments.jsonl (append-only)     │
│                                                          │
│  2. Update invest state:                                 │
│     ├── Decrement deadline                               │
│     ├── Check success (accuracy > best_before_invest)    │
│     ├── Check abort (accuracy < 70% of best)             │
│     └── Check deadline expiry                            │
│                                                          │
│  3. Enforce ideas.md size (truncate to 10 items)         │
│                                                          │
│  4. Generate analysis.txt (fixed-size, ~50 lines):       │
│     ├── Current run results                              │
│     ├── Training dynamics (trend, convergence, slope)    │
│     ├── Best experiment + rank                           │
│     ├── Last 10 experiments table                        │
│     ├── Tried configurations (1 line per dimension)      │
│     ├── Novelty score (L2 in normalized config space)    │
│     ├── Diminishing returns warnings                     │
│     └── Invest state summary                             │
│                                                          │
│  Config space dimensions (for novelty score):            │
│  ├── DEPTH        [1, 12]      linear                    │
│  ├── N_EMBD       [64, 512]    linear                    │
│  ├── LR           [1e-5, 1e-2] log                      │
│  ├── BATCH_SIZE   [2^8, 2^18]  log                      │
│  ├── DROPOUT      [0, 0.5]     linear                   │
│  ├── WEIGHT_DECAY [0, 0.5]     linear                   │
│  ├── LABEL_SMOOTH [0, 0.3]     linear                   │
│  └── WARMUP_RATIO [0, 0.2]     linear                   │
└──────────────────────────────────────────────────────────┘
```

## Invest Mechanism

```
┌──────────────────────────────────────────────────────────┐
│                   Invest State Machine                    │
│                                                          │
│  ┌─────────┐   accuracy dropped    ┌──────────────────┐ │
│  │  NORMAL ├──but foundational──> │ INVEST ACTIVE     │ │
│  │  (keep/ │   agent creates      │ deadline: 3       │ │
│  │ discard)│   invest_state.json  │ abort: 70% best   │ │
│  └────┬────┘                      └──────┬───────────┘ │
│       ^                                  │             │
│       │                    ┌─────────────┼──────────┐  │
│       │                    │             │          │  │
│       │              accuracy >     accuracy <   deadline │
│       │              best_before    threshold    expired │
│       │                    │             │          │  │
│       │                    v             v          v  │
│       │              ┌─────────┐ ┌──────────┐ ┌──────┐│
│       └──────────────┤ SUCCESS │ │  ABORT   │ │EXPIRE││
│                      │ → keep  │ │ → revert │ │→rvrt ││
│                      └─────────┘ └──────────┘ └──────┘│
│                                                        │
│  Rules:                                                │
│  - Max 1 active invest at a time                       │
│  - Default deadline: 3 experiments                     │
│  - Abort threshold: 70% of best accuracy               │
│  - analyze.py enforces all deadlines mechanically       │
└──────────────────────────────────────────────────────────┘
```

## Multi-Step Retrosynthesis (Inference)

```
retro_tree(target="caffeine", max_depth=5)
│
│  ┌────────────────────────────────────┐
│  │ model.predict("caffeine SMILES")   │
│  │ → reactant_A + reactant_B          │
│  │ (greedy or beam search)             │
│  └─────────┬──────────────┬───────────┘
│            │              │
│       ┌────▼────┐    ┌───▼────┐
│       │ react_A │    │react_B │
│       │ buyable?│    │buyable?│
│       └────┬────┘    └───┬────┘
│            │             │
│       NO   │        YES  │ → STOP (leaf)
│            │
│  ┌─────────▼──────────────────────┐
│  │ model.predict("react_A SMILES")│
│  │ → react_C + react_D            │
│  └─────────┬──────────────┬───────┘
│            │              │
│         ...recurse...   ...recurse...
│
│  Termination conditions:
│  ├── Molecule is in building blocks set → BUYABLE
│  ├── max_depth reached → STOP
│  ├── Cycle detected (same mol in ancestor path) → STOP
│  └── Invalid SMILES generated → STOP
│
│  Output: tree structure with molecules at each node,
│          buyability status at leaves, RDKit 2D drawings
```

## Frontend (Gradio)

```
┌─────────────────────────────────────────────────────────────┐
│  RetroSynth: AI-Powered Retrosynthesis Prediction           │
│  Model accuracy: 50.2% | 4 layers, 256 dim | Updated: ...  │
├────────────────────────── Tab: Predict ─────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────┐        │
│  │  Enter SMILES or molecule name:                  │        │
│  │  [                                            ]  │        │
│  │  Max depth: [===3===]  ☐ Use beam search (top-3) │        │
│  │  [Predict Route]                                 │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  Demo molecules:                                            │
│  [Aspirin] [Caffeine] [Ibuprofen] [Adipic Acid]            │
│                                                             │
│  ┌─────────┐           ┌─────────┐                          │
│  │  [img]  │  Target   │  [img]  │  Predicted reactants     │
│  └─────────┘           └─────────┘                          │
│                                                             │
│  Top Predictions (beam search):                             │
│  1. OC(=O)c1ccccc1O.CC(=O)OC(=O)C  (log-prob: -2.31)      │
│  2. OC(=O)c1ccccc1O.CC(=O)Cl       (log-prob: -3.45)      │
│  3. OC(=O)c1ccccc1O.CC(=O)O        (log-prob: -4.12)      │
│                                                             │
│  Synthesis Route (2 steps):                                 │
│  Step 1: salicylic acid + acetic anhydride --> aspirin      │
│                                                             │
├────────────── Tab: Autoresearch Progress ───────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────┐        │
│  │  Interactive Plotly chart: accuracy over expts   │        │
│  │  Green dots = kept, gray = discarded,            │        │
│  │  amber = invested, green line = running best     │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  | # | Commit  | Accuracy | Validity | Status | Desc   |   │
│  |---|---------|----------|----------|--------|--------|   │
│  | 1 | c095a7f | 0.3420   | 0.7520   | kept   | base.. |   │
│  | 2 | 33cd4fb | 0.4500   | 0.8720   | kept   | LR=..  |   │
│  | ...                                                  |   │
│                                                             │
│  ┌─────────────────────────────────────────────────┐        │
│  │  Latest training loss curve (Plotly)              │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## AWS Infrastructure (Terraform)

```
┌─ VPC ──────────────────────────────────────┐
│                                            │
│  ┌─ Public Subnet ───────────────────────┐ │
│  │                                       │ │
│  │  ┌─ Security Group ────────────────┐  │ │
│  │  │  Inbound:                       │  │ │
│  │  │  ├── SSH (22) from your IP      │  │ │
│  │  │  └── Gradio (7860) from 0.0.0.0 │  │ │
│  │  │                                 │  │ │
│  │  │  ┌─ EC2: g4dn.xlarge ────────┐ │  │ │
│  │  │  │  AMI: Deep Learning (Ubuntu)│ │  │ │
│  │  │  │  GPU: T4 16GB              │ │  │ │
│  │  │  │  CPU: 4 vCPU, 16GB RAM     │ │  │ │
│  │  │  │  Disk: 100GB gp3           │ │  │ │
│  │  │  │  Elastic IP attached       │ │  │ │
│  │  │  │  PyTorch 2.4-2.5 + CUDA124 │ │  │ │
│  │  │  └────────────────────────────┘ │  │ │
│  │  └─────────────────────────────────┘  │ │
│  └───────────────────────────────────────┘ │
│                                            │
│  Internet Gateway                          │
└────────────────────────────────────────────┘
```

## Deployment Comparison

| Feature | Local (MacBook) | Google Colab | AWS g4dn.xlarge | HF Spaces |
|---|---|---|---|---|
| GPU | None (CPU/MPS) | T4 16GB (free) | T4 16GB ($0.53/hr) | CPU (free) or GPU ($) |
| Training speed | ~35 steps/5min | ~8,400 steps/5min | ~8,400 steps/5min | N/A (inference only) |
| Agent loop | Manual only | Manual only | Autonomous (tmux) | N/A |
| Session limits | None | 90min idle, 12hr max | None | None |
| Cost | Free | Free | ~$0.53/hr | Free (CPU tier) |
| Best for | Development | Quick experiments | Overnight runs | Persistent demo |

## Device Compatibility

| Component | MacBook (local) | Colab (T4) | AWS g4dn (T4) |
|---|---|---|---|
| Device | CPU (MPS possible) | CUDA (T4) | CUDA (T4) |
| prepare.py | Full | Full | Full |
| train.py | CPU, TIME_BUDGET=60 | CUDA, TIME_BUDGET=300 | CUDA, TIME_BUDGET=300 |
| Autocast | bfloat16 (CPU) | float16 (tested at runtime) | float16 (tested at runtime) |
| torch.compile | Disabled | Disabled (T4 compat) | Disabled (T4 compat) |
| Beam search | CPU inference | GPU inference | GPU inference |
| app.py | CPU inference | GPU + share link | GPU + public IP |
| Agent loop | N/A | N/A (no SSH) | Claude Code in tmux |
