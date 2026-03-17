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
│  │program.md│             │ terraform│       │          │      │
│  └────┬─────┘             │  apply   │       └────┬─────┘      │
│       │ local test        └────┬─────┘            │ loads      │
│       │ (CPU, tiny data)       │ provisions       │ best_model │
│       v                        │                  │ .pt        │
│  Verify pipeline               │                  v            │
│  works end-to-end              │           Molecule input       │
│                                │           → retrosynthesis     │
│                                │             route display      │
└────────────────────────────────┼────────────────────────────────┘
                                 │
                    SSH / SCP    │
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                    AWS g4dn.xlarge (T4 16GB)                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  AUTORESEARCH AGENT LOOP                 │    │
│  │                                                         │    │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐          │    │
│  │  │  Modify  │───>│  Train   │───>│ Evaluate │          │    │
│  │  │ train.py │    │  5 min   │    │ accuracy │          │    │
│  │  └──────────┘    └──────────┘    └─────┬────┘          │    │
│  │       ^                                │               │    │
│  │       │          ┌─────────────────────┤               │    │
│  │       │          │                     │               │    │
│  │       │    improved?              not improved?         │    │
│  │       │          │                     │               │    │
│  │       │          v                     v               │    │
│  │       │    ┌──────────┐         ┌──────────┐           │    │
│  │       │    │   KEEP   │         │ DISCARD  │           │    │
│  │       │    │  commit  │         │git reset │           │    │
│  │       │    └──────────┘         └──────────┘           │    │
│  │       │          │                     │               │    │
│  │       │          v                     │               │    │
│  │       │    ┌──────────┐                │               │    │
│  │       │    │   Log    │<───────────────┘               │    │
│  │       └────┤results.tsv                                │    │
│  │            └──────────┘                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Outputs:                                                       │
│  ├── results.tsv          (experiment log)                      │
│  ├── best_model.pt        (best checkpoint)                     │
│  ├── loss_curve.csv       (per-experiment training curves)      │
│  └── progress.png         (autoresearch progress chart)         │
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
│  3. Build SMILES tokenizer (character-level, regex)      │
│     └── ~80-90 token vocab                               │
│     └── Special tokens: <pad>=0, <bos>=1, <eos>=2,      │
│                          <sep>=3                         │
│                                                          │
│  4. Tokenize all reactions                               │
│     └── <bos> [product] <sep> [reactants] <eos> <pad>   │
│                                                          │
│  5. Save as tensors                                      │
│     └── train.pt (40K), val.pt (5K), test.pt (5K)       │
│     └── vocab.json (token <-> id mappings)               │
│                                                          │
│  6. Download ZINC building blocks                        │
│     └── Canonical SMILES set for buyability lookup       │
│     └── building_blocks.pkl                              │
└──────────────────────────────────────────────────────────┘
```

## Model Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    GPT (Decoder-Only)                     │
│                                                          │
│  Input: <bos> C C ( = O ) O c 1 ... <sep>               │
│                                                          │
│  ┌──────────────────────────────────┐                    │
│  │  Token Embedding (vocab=~90)     │                    │
│  │  + Rotary Position Embedding     │                    │
│  └──────────────┬───────────────────┘                    │
│                 │                                        │
│  ┌──────────────▼───────────────────┐                    │
│  │  Transformer Block x4            │                    │
│  │  ┌─────────────────────────┐     │                    │
│  │  │ RMSNorm                 │     │                    │
│  │  │ Causal Self-Attention   │     │  n_embd = 256      │
│  │  │ (SDPA, n_head=4)       │     │  head_dim = 64     │
│  │  │ + Value Embedding       │     │  ~2-5M params      │
│  │  ├─────────────────────────┤     │                    │
│  │  │ RMSNorm                 │     │                    │
│  │  │ MLP (ReLU²)            │     │                    │
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
│  (product tokens masked, <pad> tokens masked)            │
└──────────────────────────────────────────────────────────┘
```

## Training Sequence Format

```
Position:  0    1  2  3  4  5  6  7  8  9  10 11 12 13  14 15 ...
Token:    <bos> C  C  (  =  O  )  O  c  1  c  c  c  c  <sep> O ...
Loss mask: 0    0  0  0  0  0  0  0  0  0  0  0  0  0   0    1 ...
           |<-- product (no loss) -->|                   |<-- reactants (loss) -->|
```

## Evaluation Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                  evaluate_retro_accuracy()                │
│                                                          │
│  For 500 val examples (batched, batch_size=64):          │
│                                                          │
│  1. Feed prefix: <bos> [product tokens] <sep>            │
│                                                          │
│  2. Greedy autoregressive generation                     │
│     └── argmax next token until <eos> or max_len         │
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
│  Metrics:                                                │
│  ├── val_accuracy = correct / total                      │
│  ├── val_validity = valid_smiles / total                 │
│  └── (secondary) partial_match = fraction of reactants   │
│       individually correct                               │
└──────────────────────────────────────────────────────────┘
```

## Multi-Step Retrosynthesis (Inference)

```
retro_tree(target="caffeine", max_depth=5)
│
│  ┌────────────────────────────────────┐
│  │ model.predict("caffeine SMILES")   │
│  │ → reactant_A + reactant_B          │
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
│  ├── Molecule is in ZINC building blocks set → BUYABLE
│  ├── max_depth reached → STOP (mark as "needs further synthesis")
│  ├── Cycle detected (same mol in ancestor path) → STOP
│  └── Invalid SMILES generated → STOP (mark as "prediction failed")
│
│  Output: tree structure with molecules at each node,
│          buyability status at leaves, RDKit 2D drawings
```

### ZINC Building Blocks Lookup

```
┌──────────────────────────────────────────────────────────┐
│              Commercially Available Check                 │
│                                                          │
│  Source: ZINC database building block catalogs            │
│  ├── Sigma-Aldrich Building Blocks                       │
│  ├── Enamine Building Blocks                             │
│  └── Combi-Blocks                                        │
│                                                          │
│  Implementation:                                         │
│  1. Download SMILES lists from ZINC catalogs             │
│  2. Canonicalize all SMILES via RDKit                    │
│  3. Store as a Python set() for O(1) lookup              │
│  4. Serialize to building_blocks.pkl                     │
│                                                          │
│  At inference time:                                      │
│  is_buyable(smiles) = canonical(smiles) in building_set  │
│                                                          │
│  Fallback (if ZINC download unavailable):                │
│  is_buyable(smiles) = mol_weight < 200 and num_rings < 3 │
└──────────────────────────────────────────────────────────┘
```

## Frontend (Gradio)

```
┌─────────────────────────────────────────────────────────────┐
│  RetroSynth: AI-Powered Retrosynthesis Prediction           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────┐        │
│  │  Enter SMILES or molecule name:                  │        │
│  │  [                                            ]  │        │
│  │  [Predict Route]                                 │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  Demo molecules:                                            │
│  [Aspirin] [Caffeine] [Ibuprofen] [Adipic Acid]            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Target: Caffeine                                           │
│  ┌─────────┐                                                │
│  │  [img]  │  Cn1c(=O)c2c(ncn2C)n(C)c1=O                  │
│  └─────────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  Step 1: Methylation                                        │
│  ┌─────────┐     ┌─────────┐                                │
│  │  [img]  │  +  │  [img]  │                                │
│  │ react A │     │ react B │                                │
│  └────┬────┘     └─────────┘                                │
│       │           [BUYABLE]                                 │
│       ▼                                                     │
│  Step 2: ...                                                │
│  ┌─────────┐     ┌─────────┐                                │
│  │  [img]  │  +  │  [img]  │                                │
│  └─────────┘     └─────────┘                                │
│  [BUYABLE]       [BUYABLE]                                  │
│                                                             │
│  Route complete in 2 steps from commercial materials.       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Autoresearch Progress                                      │
│  ┌─────────────────────────────────────────────────┐        │
│  │ val_accuracy chart (embedded progress.png)       │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  Model: 36 experiments, best accuracy: 31.2%                │
│  Training loss curves: [Show/Hide]                          │
└─────────────────────────────────────────────────────────────┘
```

## Results Logging

### results.tsv (per experiment)
```
commit    val_accuracy    val_validity    memory_gb    status    description
a1b2c3d   0.120000        0.850000        2.1          keep      baseline
b2c3d4e   0.185000        0.890000        2.1          keep      increase LR to 0.04
c3d4e5f   0.105000        0.820000        2.1          discard   switch to GeLU
d4e5f6g   0.000000        0.000000        0.0          crash     double depth (OOM)
```

### loss_curve.csv (per experiment, overwritten each run)
```
step,loss,experiment_id
0,4.512,b2c3d4e
10,3.891,b2c3d4e
20,3.214,b2c3d4e
...
```

## Visualizations (3 charts)

### Chart 1: Autoresearch Progress (across experiments)
- X: experiment number
- Y: val_accuracy (higher is better)
- Green dots: kept improvements
- Gray dots: discarded attempts
- Green staircase: running best (cummax)
- **Story: "The agent autonomously improved retrosynthesis accuracy over N experiments"**

### Chart 2: Training Loss Curves (within experiments)
- X: training step
- Y: cross-entropy loss
- Multiple overlaid curves (one per experiment, color-coded)
- **Story: "Different training configurations produce different learning dynamics"**

### Chart 3: Retrosynthesis Route (frontend)
- Tree of molecules with 2D structure drawings
- Arrows showing synthetic direction
- Buyability labels on leaves
- **Story: "Here's what the model can actually do"**

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
│  │  │  └────────────────────────────┘ │  │ │
│  │  └─────────────────────────────────┘  │ │
│  └───────────────────────────────────────┘ │
│                                            │
│  Internet Gateway                          │
└────────────────────────────────────────────┘

Estimated cost: ~$0.55/hr ($3.30 for 6 hours)
```

## Google Colab Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Colab (Free T4 GPU)                │
│                                                             │
│  colab.ipynb                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Cell 1: !nvidia-smi (verify GPU)                   │    │
│  │  Cell 2: Install uv                                 │    │
│  │  Cell 3: git clone repo                             │    │
│  │  Cell 4: uv sync                                    │    │
│  │  Cell 5: uv run prepare.py (~2 min)                 │    │
│  │  Cell 6: uv run train.py (~5 min)                   │    │
│  │  Cell 7: Plot loss curve                            │    │
│  │  Cell 8: uv run app.py (Gradio with share link)     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Limitations:                                               │
│  ├── No SSH (free tier blocks it)                           │
│  ├── Cannot run Claude Code autonomously                    │
│  ├── Session disconnects after ~90 min idle                 │
│  ├── Max 12 hour session                                    │
│  └── Manual experiments only (no agent loop)                │
│                                                             │
│  Best for: quick testing, one-off experiments, demos        │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Comparison

| Feature | Local (MacBook) | Google Colab | AWS g4dn.xlarge |
|---|---|---|---|
| GPU | None (CPU/MPS) | T4 16GB (free) | T4 16GB ($0.53/hr) |
| Training speed | ~35 steps/5min | ~8,400 steps/5min | ~8,400 steps/5min |
| Agent loop | Manual only | Manual only | Autonomous (tmux + Claude Code) |
| Session limits | None | 90min idle, 12hr max | None |
| Cost | Free | Free | ~$0.53/hr |
| Data persistence | Local disk | Lost on disconnect | EBS (persists across stop/start) |
| Best for | Development, testing | Quick experiments, demos | Overnight autonomous runs |

## Device Compatibility

| Component | MacBook Air M3 (local) | Google Colab (T4) | AWS g4dn.xlarge (T4) |
|---|---|---|---|
| Device | CPU (MPS possible) | CUDA (T4) | CUDA (T4) |
| prepare.py | Full | Full | Full |
| train.py | CPU, TIME_BUDGET=60 | CUDA, TIME_BUDGET=300 | CUDA, TIME_BUDGET=300 |
| Attention | SDPA (CPU) | SDPA (CUDA) | SDPA (CUDA) |
| torch.compile | Disabled | Disabled (T4 compat) | Disabled (T4 compat) |
| autocast | bfloat16 (CPU) | Disabled (CUBLAS issue) | Disabled (CUBLAS issue) |
| app.py | CPU inference | GPU inference + share link | GPU inference + public IP |
| Agent loop | N/A | N/A (no SSH) | Claude Code in tmux |

## Known Compatibility Issues

### T4 + PyTorch 2.10 + CUDA 12.8
- `torch.compile` with `dynamic=False` crashes on T4 (not enough SMs)
- `torch.amp.autocast` with float16 triggers CUBLAS_STATUS_INVALID_VALUE
- **Workaround**: agent disabled both, trains in float32 (~2x slower)
- **Fix**: use PyTorch 2.4-2.5 or switch to A10G/L4 instance (supports bfloat16)

### Colab Nested Directory
- `git clone` creates `AutoRetroSynthesis/AutoRetroSynthesis/` nesting
- `colab.ipynb` auto-detects and `cd`s to correct directory
- Caused by repo structure on GitHub
