
#!/usr/bin/env python3
"""
collect_results.py
------------------
Scan one or more run directories for model outputs and aggregate them into a single CSV.

It looks for:
  - "*_test.json" (final metrics)
  - matching "*_args.json" (run configuration)

It writes a CSV with columns like:
  dataset, layer, model, backend, emb_dim, epochs, batch_size, lr, acc, f1_macro, auc_approx, test_loss, run_dir, run_name

Usage:
  python scripts/collect_results.py --runs ./results_sle ./results_gat ./results_mell --out all_results.csv

Tips:
  - You can pass the parent results folder; the script will crawl subfolders recursively.
  - Works with outputs produced by lpmp.cli sle/gat/mell/hoplp/rmne and the sweep/random/optuna/all-layers runners.

"""

import argparse, os, json, csv
from glob import glob
from pathlib import Path

def _safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def find_runs(root: str):
    """Yield (args_path, test_path) pairs discovered under root."""
    for test in glob(os.path.join(root, "**", "*_test.json"), recursive=True):
        # Find a matching args file in the same directory (same prefix if present)
        d = os.path.dirname(test)
        base = os.path.basename(test).replace("_test.json", "")
        # preferred exact match
        args_path = os.path.join(d, f"{base}_args.json")
        if not os.path.exists(args_path):
            # fallback: any args.json in the same folder
            cand = glob(os.path.join(d, "*_args.json"))
            args_path = cand[0] if cand else None
        yield (args_path, test)

def infer_row(args_path, test_path):
    # Defaults
    row = {
        "dataset": None,
        "layer": None,
        "model": None,
        "backend": None,
        "emb_dim": None,
        "epochs": None,
        "batch_size": None,
        "lr": None,
        "acc": None,
        "f1_macro": None,
        "auc_approx": None,
        "test_loss": None,
        "run_dir": str(Path(test_path).parent),
        "run_name": Path(test_path).name.replace("_test.json","")
    }

    # Load test metrics
    try:
        with open(test_path) as f:
            te = json.load(f)
        row["acc"] = _safe_get(te, "acc")
        row["f1_macro"] = _safe_get(te, "f1_macro")
        row["auc_approx"] = _safe_get(te, "auc_approx")
        row["test_loss"] = _safe_get(te, "test_loss")
    except Exception:
        pass

    # Load args/config if available
    args = None
    if args_path and os.path.exists(args_path):
        try:
            with open(args_path) as f:
                args = json.load(f)
        except Exception:
            args = None

    # Extract fields from args
    if args:
        # dataset name from edges path parent folder if available
        edges = _safe_get(args, "edges")
        if edges:
            # e.g., data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges -> CS-Aarhus_multiplex
            dataset = Path(edges).parent.name
            row["dataset"] = dataset

        row["layer"] = _safe_get(args, "target_layer")
        row["epochs"] = _safe_get(args, "epochs")
        row["batch_size"] = _safe_get(args, "batch_size")
        row["lr"] = _safe_get(args, "lr")

        # model command (sle/gat/mell/hoplp/rmne) if present
        # When saved by lpmp.cli, we included args dict so "cmd" might not be there;
        # infer from run_name as fallback.
        row["model"] = _safe_get(args, "cmd")
        if row["model"] is None:
            rn = row["run_name"].lower()
            for m in ["sle","gat","mell","hoplp","rmne"]:
                if rn.startswith(m) or f"_{m}_" in rn:
                    row["model"] = m
                    break

        # embedding backend (either file or computed)
        if "embed-backend" in args:
            row["backend"] = args.get("embed-backend")
        elif "embed_backend" in args:
            row["backend"] = args.get("embed_backend")
        else:
            row["backend"] = "precomputed" if _safe_get(args, "node2vec") else None

        # embedding dimension
        row["emb_dim"] = _safe_get(args, "emb_dim")
        if row["emb_dim"] is None:
            # may be deduced as 2*d_node in GAT/SLE, but we keep raw embedding dim here
            pass

    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="One or more root directories to scan recursively")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    rows = []
    for root in args.runs:
        for (apath, tpath) in find_runs(root):
            rows.append(infer_row(apath, tpath))

    # stable header
    fields = ["dataset","layer","model","backend","emb_dim","epochs","batch_size","lr","acc","f1_macro","auc_approx","test_loss","run_dir","run_name"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
