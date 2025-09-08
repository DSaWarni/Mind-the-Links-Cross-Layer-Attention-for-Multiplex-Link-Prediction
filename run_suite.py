#!/usr/bin/env python3
"""
run_suite.py
------------
Run SLE/GAT/MeLL/RMNE/HOPLP-MUL across ALL layers of a dataset, sweeping a grid of hyperparameters
and multiple seeds. Select the BEST hyperparameter config per layer based on mean **validation F1**
across seeds, and summarize TEST metrics for that best config.

Usage (single line examples):
  python run_suite.py --model sle --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --seeds 42,43,44 --outroot ./suite_results --early-stopping 10 --epochs 100 --batch-size 512
  python run_suite.py --model all --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges --seeds 42,43 --outroot ./suite_results

Optional: provide a JSON grid file to override defaults (see --print-default-grid).
  python run_suite.py --model sle --edges ... --grid sle_grid.json --seeds 42,43 --outroot ./suite_results

The JSON file should be a list of dicts; each dict contains CLI flags (without --edges, --target-layer, --outdir, --seed).
Example sle_grid.json:
[
  {"--embed-backend":"node2vec","--emb-dim":128,"--n2v-p":1.0,"--n2v-q":1.0},
  {"--embed-backend":"node2vec","--emb-dim":256,"--n2v-p":0.5,"--n2v-q":2.0}
]
"""

import argparse, subprocess, sys, os, json, csv, math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# ---------- Helpers ----------

def list_layers_from_edges(edges_path: str) -> List[str]:
    # Use lpmp.data to parse layers robustly
    from lpmp.data import load_edges
    d = load_edges(edges_path)
    return list(d.layers)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: List[str], cwd: str = None):
    print("[RUN]", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")

def read_best_val_f1(csv_path: str) -> float:
    # training_log.csv may be per-epoch (SLE/GAT/MeLL/RMNE) or summary (HOPLP)
    # We search all rows for maximum val_f1/val_f1_macro.
    if not Path(csv_path).exists():
        return float("nan")
    best = -1.0
    with open(csv_path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            # try multiple field names
            for key in ("val_f1","val_f1_macro"):
                if key in row and row[key] not in (None,"","NaN"):
                    try:
                        val = float(row[key])
                        if val > best:
                            best = val
                    except:
                        pass
    return best

def read_test_metrics(json_path: str) -> Dict[str, float]:
    if not Path(json_path).exists():
        return {}
    with open(json_path) as fh:
        j = json.load(fh)
    # normalize keys
    out = {}
    for k in ("f1_macro","acc","auc_approx","test_loss","threshold"):
        if k in j:
            out[k] = j[k]
    # some baselines use prefixed keys
    for alt, std in (("test_f1_macro","f1_macro"),("test_acc","acc"),("test_auc_approx","auc_approx")):
        if alt in j and std not in out:
            out[std] = j[alt]
    return out

def default_grid(model: str) -> List[Dict[str, Any]]:
    if model == "sle":
        return [
            {"--embed-backend": "node2vec", "--emb-dim": 128, "--n2v-p": 1.0, "--n2v-q": 1.0, "--n2v-walk-len": 40, "--n2v-walks-per-node": 10, "--sgns-window": 10, "--sgns-epochs": 1, "--sgns-batch": 8192, "--sgns-lr": 0.001, "--sgns-neg-k": 1, "--sle-layers": 2, "--sle-nhead": 4, "--sle-ff": 512, "--sle-dropout": 0.1},
            {"--embed-backend": "node2vec", "--emb-dim": 256, "--n2v-p": 0.5, "--n2v-q": 2.0, "--n2v-walk-len": 80, "--n2v-walks-per-node": 20, "--sgns-window": 10, "--sgns-epochs": 1, "--sgns-batch": 8192, "--sgns-lr": 0.001, "--sgns-neg-k": 3, "--sle-layers": 3, "--sle-nhead": 8, "--sle-ff": 512, "--sle-dropout": 0.2},
            {"--embed-backend": "node2vec", "--emb-dim": 64, "--n2v-p": 2.0, "--n2v-q": 0.5, "--n2v-walk-len": 20, "--n2v-walks-per-node": 5, "--sgns-window": 5, "--sgns-epochs": 2, "--sgns-batch": 4096, "--sgns-lr": 0.0005, "--sgns-neg-k": 5, "--sle-layers": 1, "--sle-nhead": 2, "--sle-ff": 256, "--sle-dropout": 0.0},
            {"--embed-backend": "corewalk", "--emb-dim": 128, "--core-temp": 1.0, "--core-walk-len": 40, "--core-walks-per-node": 10, "--sgns-window": 10, "--sgns-epochs": 1, "--sgns-batch": 8192, "--sgns-lr": 0.001, "--sgns-neg-k": 1, "--sle-layers": 2, "--sle-nhead": 4, "--sle-ff": 512, "--sle-dropout": 0.1},
            {"--embed-backend": "corewalk", "--emb-dim": 64, "--n2v-p": 1.0, "--n2v-q": 2.0, "--n2v-walk-len": 60, "--n2v-walks-per-node": 15, "--sgns-window": 10, "--sgns-epochs": 2, "--sgns-batch": 16384, "--sgns-lr": 0.0015, "--sgns-neg-k": 2, "--sle-layers": 2, "--sle-nhead": 4, "--sle-ff": 768, "--sle-dropout": 0.15}
        ]
    if model == "gat":
        return [
            {"--embed-backend": "node2vec","--emb-dim": 128,"--n2v-p": 1.0,"--n2v-q": 1.0,"--n2v-walk-len": 40,"--n2v-walks-per-node": 10,"--sgns-window": 10,"--sgns-epochs": 1,"--sgns-batch": 8192,"--sgns-lr": 0.001,"--sgns-neg-k": 1,"--gat-heads": 4,"--gat-dropout": 0.1,"--tfm-layers": 2,"--tfm-heads": 4,"--tfm-ff": 512,"--tfm-dropout": 0.1},
            {"--embed-backend": "node2vec","--emb-dim": 256,"--n2v-p": 0.5,"--n2v-q": 2.0,"--n2v-walk-len": 80,"--n2v-walks-per-node": 20,"--sgns-window": 10,"--sgns-epochs": 1,"--sgns-batch": 8192,"--sgns-lr": 0.001,"--sgns-neg-k": 3,"--gat-heads": 8,"--gat-dropout": 0.2,"--tfm-layers": 2,"--tfm-heads": 4,"--tfm-ff": 512,"--tfm-dropout": 0.1},
            {"--embed-backend": "corewalk","--emb-dim": 128,"--core-temp": 1.0,"--core-walk-len": 40,"--core-walks-per-node": 10,"--sgns-window": 10,"--sgns-epochs": 1,"--sgns-batch": 8192,"--sgns-lr": 0.001,"--sgns-neg-k": 1,"--gat-heads": 2,"--gat-dropout": 0.1,"--tfm-layers": 3,"--tfm-heads": 4,"--tfm-ff": 768,"--tfm-dropout": 0.1},
            {"--embed-backend": "node2vec","--emb-dim": 64,"--n2v-p": 2.0,"--n2v-q": 0.5,"--n2v-walk-len": 20,"--n2v-walks-per-node": 5,"--sgns-window": 5,"--sgns-epochs": 2,"--sgns-batch": 4096,"--sgns-lr": 0.0005,"--sgns-neg-k": 5,"--gat-heads": 2,"--gat-dropout": 0.0,"--tfm-layers": 1,"--tfm-heads": 2,"--tfm-ff": 256,"--tfm-dropout": 0.0},
            {"--embed-backend": "corewalk","--emb-dim": 64,"--n2v-p": 1.0,"--n2v-q": 2.0,"--n2v-walk-len": 60,"--n2v-walks-per-node": 15,"--sgns-window": 10,"--sgns-epochs": 2,"--sgns-batch": 16384,"--sgns-lr": 0.0015,"--sgns-neg-k": 2,"--gat-heads": 4,"--gat-dropout": 0.15,"--tfm-layers": 2,"--tfm-heads": 8,"--tfm-ff": 768,"--tfm-dropout": 0.15}
        ]
    if model == "mell":
        return [
            { "--dim": 64,  "--lr": 0.0015 },
            { "--dim": 128, "--lr": 0.0010 },
            { "--dim": 192, "--lr": 0.0010 },
            { "--dim": 256, "--lr": 0.0005 },
            { "--dim": 128, "--lr": 0.0005 }
        ]
    if model == "rmne":
        return [
            { "--dim": 64,  "--K": 10, "--Le": 40, "--window": 5,  "--lr": 0.0015, "--r": 4.0, "--t": 6.0 },
            { "--dim": 128, "--K": 10, "--Le": 40, "--window": 5,  "--lr": 0.0010, "--r": 5.0, "--t": 5.0 },
            { "--dim": 192, "--K": 15, "--Le": 60, "--window": 7,  "--lr": 0.0010, "--r": 5.0, "--t": 5.0 },
            { "--dim": 256, "--K": 20, "--Le": 80, "--window": 10, "--lr": 0.0005, "--r": 6.0, "--t": 6.0 },
            { "--dim": 128, "--K": 20, "--Le": 60, "--window": 10, "--lr": 0.0008, "--r": 5.0, "--t": 7.0 }
        ]
    if model == "hoplp":
        return [
            { "--K": 3, "--alpha": 0.5, "--weight-mode": "density" },
            { "--K": 5, "--alpha": 0.7, "--weight-mode": "uniform" },
            { "--K": 4, "--alpha": 0.4, "--weight-mode": "density" },
            { "--K": 2, "--alpha": 0.6, "--weight-mode": "uniform" },
            { "--K": 5, "--alpha": 0.3, "--weight-mode": "density" }
        ]
    raise ValueError(model)

def model_subcmd(model: str) -> str:
    return {"sle":"sle","gat":"gat","mell":"mell","rmne":"rmne","hoplp":"hoplp"}[model]

def build_cmd(base: List[str], flags: Dict[str, Any]) -> List[str]:
    cmd = list(base)
    for k, v in flags.items():
        if isinstance(v, bool):
            if v: cmd.append(k)  # only include if True
        else:
            cmd.extend([k, str(v)])
    return cmd

def summarize_and_choose(best_root: Path, model: str, layer: str, seeds: List[int], cfgs: List[Dict[str,Any]], out_summary_dir: Path):
    # For each config, average best VAL F1 across seeds; choose best config; also aggregate TEST metrics for that config.
    val_means = []; val_stds = []
    for i, _ in enumerate(cfgs):
        vals = []
        for sd in seeds:
            run_dir = best_root / f"cfg{i}" / f"seed{sd}"
            csv_path = run_dir / "training_log.csv"
            best_val = read_best_val_f1(str(csv_path))
            if not math.isnan(best_val):
                vals.append(best_val)
        m = float(np.mean(vals)) if vals else float("nan")
        s = float(np.std(vals)) if vals else float("nan")
        val_means.append(m); val_stds.append(s)

    # choose best config by highest mean val F1
    best_idx = int(np.nanargmax(val_means)) if any([not math.isnan(x) for x in val_means]) else 0

    # aggregate test metrics for chosen config
    tests = {"f1_macro": [], "acc": [], "auc_approx": []}
    for sd in seeds:
        run_dir = best_root / f"cfg{best_idx}" / f"seed{sd}"
        # test json name varies by runner; search for *_test.json
        test_files = list(run_dir.glob("*_test.json")) + list(run_dir.glob("*.json"))
        jf = None
        for cand in test_files:
            if "test" in cand.name:
                jf = cand; break
        if jf is None: 
            continue
        metrics = read_test_metrics(str(jf))
        for k in tests.keys():
            if k in metrics:
                tests[k].append(float(metrics[k]))

    # compute means/stds
    out = {
        "model": model,
        "layer": layer,
        "best_cfg_index": best_idx,
        "best_cfg_params": cfgs[best_idx],
        "val_f1_mean": val_means[best_idx],
        "val_f1_std": val_stds[best_idx],
        "test_f1_mean": float(np.mean(tests["f1_macro"])) if tests["f1_macro"] else float("nan"),
        "test_f1_std": float(np.std(tests["f1_macro"])) if tests["f1_macro"] else float("nan"),
        "test_acc_mean": float(np.mean(tests["acc"])) if tests["acc"] else float("nan"),
        "test_auc_mean": float(np.mean(tests["auc_approx"])) if tests["auc_approx"] else float("nan"),
        "seeds": seeds,
    }

    # write per-layer summary
    ensure_dir(str(out_summary_dir))
    with open(out_summary_dir / f"{model}_layer_{layer}_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)

    return out

def write_master_csv(rows: List[Dict[str,Any]], out_csv: Path):
    ensure_dir(str(out_csv.parent))
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model","layer","best_cfg_index","best_cfg_params","val_f1_mean","val_f1_std","test_f1_mean","test_f1_std","test_acc_mean","test_auc_mean","seeds"])
        for r in rows:
            w.writerow([
                r["model"], r["layer"], r["best_cfg_index"], json.dumps(r["best_cfg_params"]),
                r["val_f1_mean"], r["val_f1_std"], r["test_f1_mean"], r["test_f1_std"],
                r["test_acc_mean"], r["test_auc_mean"], ",".join(map(str, r["seeds"]))
            ])

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sle","gat","mell","rmne","hoplp","all"])
    ap.add_argument("--edges", required=True, type=str)
    ap.add_argument("--layers", type=str, default="all", help="Comma-separated list or 'all'")
    ap.add_argument("--seeds", type=str, default="42", help="Comma-separated list, e.g., 42,43,44")
    ap.add_argument("--grid", type=str, default=None, help="JSON file containing list of flag dicts; if empty, use defaults")
    ap.add_argument("--outroot", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--early-stopping", type=int, default=10)
    ap.add_argument("--csv-log", action="store_true")
    ap.add_argument("--print-default-grid", action="store_true")
    args, unknown = ap.parse_known_args()

    models = ["sle","gat","mell","rmne","hoplp"] if args.model == "all" else [args.model]

    # layers
    if args.layers == "all":
        layers = list_layers_from_edges(args.edges)
    else:
        layers = [s.strip() for s in args.layers.split(",") if s.strip()]

    # seeds
    seeds = [int(s) for s in args.seeds.split(",")]

    # grids
    grids = {}
    if args.grid:
        with open(args.grid) as fh:
            grids_for_single = json.load(fh)
        # same grid for chosen model
        grids = {m: grids_for_single for m in models}
    else:
        for m in models:
            grids[m] = default_grid(m)

    if args.print_default_grid:
        print(json.dumps(grids, indent=2))
        sys.exit(0)

    # Master results
    master_rows = []

    for m in models:
        sub = model_subcmd(m)
        cfgs = grids[m]

        for layer in layers:
            # run each cfg x seed
            for i, cfg in enumerate(cfgs):
                for sd in seeds:
                    outdir = Path(args.outroot) / m / f"L{layer}" / f"cfg{i}" / f"seed{sd}"
                    ensure_dir(str(outdir))

                    base = ["python","-m","lpmp.cli", sub,
                            "--edges", args.edges,
                            "--target-layer", str(layer),
                            "--epochs", str(args.epochs),
                            "--batch-size", str(args.batch_size),
                            "--outdir", str(outdir),
                            "--early-stopping", str(args.early_stopping),
                            "--seed", str(sd)]
                    if args.csv_log:
                        base.append("--csv-log")

                    # HOPLP has no epoch-wise learning but accepts these flags harmlessly
                    cmd = build_cmd(base, cfg)

                    try:
                        run_cmd(cmd)
                    except Exception as e:
                        print(f"[WARN] Failed run: model={m} layer={layer} cfg={i} seed={sd}: {e}", file=sys.stderr)

            # summarize and choose best config by mean val F1 over seeds
            best_root = Path(args.outroot) / m / f"L{layer}"
            summary_dir = Path(args.outroot) / "summaries"
            row = summarize_and_choose(best_root, m, str(layer), seeds, cfgs, summary_dir)
            master_rows.append(row)

    # write master CSV
    write_master_csv(master_rows, Path(args.outroot) / "summaries" / "master_summary.csv")
    print(f"[DONE] Wrote summaries to {Path(args.outroot) / 'summaries'}")

if __name__ == "__main__":
    main()
