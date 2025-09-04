# Multiplex Link Prediction (mp_lp)

This repository implements **link prediction in multiplex networks** using:
- **SLE Transformer**
- **GAT Transformer**
- Faithful baselines: **MeLL**, **RMNE**, and **HOPLP-MUL**.

All models support training, validation, and testing with **patience-based early stopping** for fair comparison.

---

## 📂 Datasets
Supported multiplex datasets (edges format: `Layer Start_Node End_Node Edge_Weight`):
- `CS-Aarhus_multiplex`
- `Vickers`
- `CKM_physicians`
- `Rattus_genetics`
- (and more, in the same format)

---

## ⚙️ Installation
```bash
conda create -n gnn python=3.12 -y
conda activate gnn
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run SLE Transformer
```bash
python -m lpmp.cli sle --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --embed-backend node2vec --emb-dim 128 --target-layer 1 --epochs 100 --batch-size 512 --outdir ./results_sle_arhus --early-stopping 10 --csv-log
```

### 2. Run GAT Transformer
```bash
python -m lpmp.cli gat --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --embed-backend node2vec --emb-dim 128 --target-layer 2 --epochs 100 --batch-size 512 --outdir ./results_gat_arhus --early-stopping 10 --csv-log
```

### 3. Run Baselines (Faithful Implementations)

#### MeLL
```bash
python -m lpmp.cli mell --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --epochs 100 --batch-size 512 --outdir ./results_mell_arhus --early-stopping 10 --csv-log --weight-decay 1e-4
```

#### RMNE
```bash
python -m lpmp.cli rmne --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --dim 128 --K 10 --Le 40 --window 5 --epochs 100 --batch-size 512 --outdir ./results_rmne_arhus --early-stopping 10 --csv-log --weight-decay 1e-4
```

#### HOPLP-MUL
```bash
python -m lpmp.cli hoplp --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --outdir ./results_hoplp_arhus
```

---

## 📊 Dataset-Specific Examples

### CS Aarhus (multiplex social network)
```bash
python -m lpmp.cli sle --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --embed-backend node2vec --emb-dim 128 --target-layer 1 --epochs 50 --batch-size 512 --outdir ./results_sle_arhus --early-stopping 10 --csv-log
```

### Vickers
```bash
python -m lpmp.cli sle --edges ./data_files/Vickers/Vickers.edges --embed-backend node2vec --emb-dim 128 --target-layer 1 --epochs 50 --batch-size 512 --outdir ./results_sle_vickers --early-stopping 10 --csv-log
```

### CKM Physicians
```bash
python -m lpmp.cli sle --edges ./data_files/CKM_physicians/CKM_physicians.edges --embed-backend node2vec --emb-dim 128 --target-layer 1 --epochs 50 --batch-size 512 --outdir ./results_sle_ckm --early-stopping 10 --csv-log
```

### Rattus Genetics
```bash
python -m lpmp.cli sle --edges ./data_files/Rattus_genetics/Rattus_genetics.edges --embed-backend node2vec --emb-dim 128 --target-layer 1 --epochs 50 --batch-size 512 --outdir ./results_sle_rattus --early-stopping 10 --csv-log
```

(Replace `sle` with `gat`, `mell`, `rmne`, or `hoplp` to run other methods.)

---

## 📈 Output
Each run produces in `--outdir`:
- Per-epoch logs (`training_log.csv` if `--csv-log` is set)
- Saved args (`*_args.json`)
- Final test metrics (`*_test.json`)

---

## 🔍 Notes
- MeLL and RMNE are now **faithful to their original papers**, with **epoch-wise training and patience**.
- HOPLP-MUL remains a **non-iterative baseline** (single run).

---

## ✅ Example Comparison Workflow
To compare all methods on CS Aarhus:
```bash
python -m lpmp.cli sle   --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --epochs 50 --batch-size 512 --outdir ./results_sle_arhus --early-stopping 10 --csv-log
python -m lpmp.cli gat   --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --epochs 50 --batch-size 512 --outdir ./results_gat_arhus --early-stopping 10 --csv-log
python -m lpmp.cli mell  --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --epochs 50 --batch-size 512 --outdir ./results_mell_arhus --early-stopping 10 --csv-log
python -m lpmp.cli rmne  --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --epochs 50 --batch-size 512 --outdir ./results_rmne_arhus --early-stopping 10 --csv-log
python -m lpmp.cli hoplp --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --outdir ./results_hoplp_arhus
```

---

## 🔬 Parameter Sensitivity Analysis (Suggested Ranges & One‑Liners)

This section lists **safe sweep ranges** that keep the math intact and are comparable across methods.  
Always keep **splits fixed** during sweeps (`--splits <file>.json`) to avoid variance from different partitions.

### Common flags to keep constant in sweeps
- `--edges` (dataset path)
- `--target-layer` (evaluate one layer at a time)
- `--splits` (use a frozen split JSON for fair comparison)
- `--mask-test-in-target` (recommended for leakage‑free baselines during unsupervised prep)

> Create frozen splits once (recommended):
```bash
python -m lpmp.cli freeze-splits --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --out ./splits_aarhus_L1.json
```

### 1) SLE / GAT (depend on an embedding backend)

**Embeddings (Node2Vec)**
- `--emb-dim`: **64, 128, 256**
- `--n2v-walk-len`: **20, 40, 80**
- `--n2v-walks-per-node`: **5, 10, 20**
- `--n2v-p` (return) and `--n2v-q` (in‑out): explore **{0.5, 1.0, 2.0}**
- SGNS: `--sgns-window`: **5, 10**, `--sgns-epochs`: **1–3**, `--sgns-batch`: **4096–16384**, `--sgns-lr`: **5e‑4–2e‑3**, `--sgns-neg-k`: **1–5**

**Model**
- SLE: `--sle-layers`: **1–3**, `--sle-nhead`: **2–8**, `--sle-ff`: **256–1024**, `--sle-dropout`: **0.0–0.3**
- GAT: `--gat-heads`: **2–8**, `--gat-dropout`: **0.0–0.3**; Transformer: `--tfm-layers`: **1–3**, `--tfm-heads`: **2–8**, `--tfm-ff`: **256–1024**, `--tfm-dropout`: **0.0–0.3**

**Optimization**
- `--lr`: **5e‑4–2e‑3**, `--weight-decay`: **1e‑5–1e‑3**, `--batch-size`: **256–1024**, `--early-stopping`: **5–20**

**Examples (single‑line):**
```bash
python -m lpmp.cli sle --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --embed-backend node2vec --emb-dim 128 --n2v-p 0.5 --n2v-q 2.0 --n2v-walk-len 80 --n2v-walks-per-node 20 --sgns-window 10 --sgns-epochs 2 --sgns-batch 8192 --sgns-lr 0.001 --sgns-neg-k 3 --target-layer 1 --epochs 100 --batch-size 512 --outdir ./sweeps/sle_a_01 --splits ./splits_aarhus_L1.json --mask-test-in-target --early-stopping 10 --csv-log
```
```bash
python -m lpmp.cli gat --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --embed-backend node2vec --emb-dim 256 --n2v-p 1.0 --n2v-q 1.0 --n2v-walk-len 40 --n2v-walks-per-node 10 --sgns-window 10 --sgns-epochs 1 --sgns-batch 8192 --sgns-lr 0.001 --sgns-neg-k 1 --gat-heads 8 --gat-dropout 0.2 --tfm-layers 2 --tfm-heads 4 --tfm-ff 512 --tfm-dropout 0.1 --target-layer 1 --epochs 100 --batch-size 512 --outdir ./sweeps/gat_a_01 --splits ./splits_aarhus_L1.json --mask-test-in-target --early-stopping 10 --csv-log
```

> To test **Core2Vec** instead, replace the Node2Vec flags with: `--embed-backend corewalk --core-temp 1.0 --core-walk-len 40 --core-walks-per-node 10`.

---

### 2) MeLL (faithful)

- Representation dim: `--dim`: **64, 128, 256**
- Optimization: `--lr`: **5e‑4–2e‑3**, `--batch-size`: **256–1024**, `--early-stopping`: **5–20**
- (All training is now epoch‑wise with patience on **val macro‑F1**; threshold is selected on validation each epoch and frozen for test.)

**Examples:**
```bash
python -m lpmp.cli mell --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --dim 128 --epochs 100 --batch-size 512 --outdir ./sweeps/mell_a_01 --splits ./splits_aarhus_L1.json --mask-test-in-target --early-stopping 10 --csv-log
```
```bash
python -m lpmp.cli mell --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --dim 256 --lr 0.0005 --epochs 150 --batch-size 512 --outdir ./sweeps/mell_a_02 --splits ./splits_aarhus_L1.json --mask-test-in-target --early-stopping 15 --csv-log
```

---

### 3) RMNE (faithful)

- Embedding dim: `--dim`: **64, 128, 256**
- Walks: `--K`: **5–20**, `--Le`: **20–80**
- Skip‑gram window: `--window`: **3–10**
- Role hyperparams: `--r`, `--t`: **3–8** (keep symmetric to start)
- Optimization: `--lr`: **5e‑4–2e‑3**, `--batch-size`: **512–4096**, `--early-stopping`: **5–20**

**Examples:**
```bash
python -m lpmp.cli rmne --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --dim 128 --K 10 --Le 40 --window 5 --lr 0.001 --epochs 100 --batch-size 2048 --outdir ./sweeps/rmne_a_01 --splits ./splits_aarhus_L1.json --mask-test-in-target --early-stopping 10 --csv-log
```
```bash
python -m lpmp.cli rmne --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --dim 256 --K 20 --Le 80 --window 10 --lr 0.0005 --epochs 150 --batch-size 4096 --outdir ./sweeps/rmne_a_02 --splits ./splits_aarhus_L1.json --mask-test-in-target --early-stopping 15 --csv-log
```

---

### 4) HOPLP‑MUL (faithful, non‑iterative)

HOPLP‑MUL computes higher‑order path similarities and fuses layers — **no epoch‑wise learning**.  
Sweep only **structural choices**:
- `--K`: **2–5**
- `--alpha`: **0.3–0.7**
- `--weight-mode`: `density` or `uniform`

**Examples:**
```bash
python -m lpmp.cli hoplp --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --K 3 --alpha 0.5 --weight-mode density --outdir ./sweeps/hoplp_a_01 --splits ./splits_aarhus_L1.json --mask-test-in-target
```
```bash
python -m lpmp.cli hoplp --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --target-layer 1 --K 5 --alpha 0.7 --weight-mode uniform --outdir ./sweeps/hoplp_a_02 --splits ./splits_aarhus_L1.json --mask-test-in-target
```

---

### Repro tips
- Set `--seed` (default 42) and **freeze splits**.
- Keep **threshold selection** on validation (done automatically).
- For cross‑dataset comparability, keep `--epochs`, `--early-stopping`, and logging identical across methods.
