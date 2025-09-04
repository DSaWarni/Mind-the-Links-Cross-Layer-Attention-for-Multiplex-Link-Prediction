
"""
All-layers Bayesian optimization runner.

For each layer:
  1) Run Optuna (TPE) to tune hyperparameters for N trials
  2) Re-run the CLI once using the best params to produce final test metrics
  3) Append a consolidated CSV row

Example (SLE):
    python scripts/all_layers_optuna.py           --mode sle           --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges           --node2vec ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl           --outdir ./results_sle_all_optuna           --n-trials 30 --epochs 30 --seed 42

Example (GAT):
    python scripts/all_layers_optuna.py           --mode gat           --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges           --node2vec ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl           --outdir ./results_gat_all_optuna           --n-trials 30 --epochs 30 --seed 42
"""
import argparse, os, sys, json, subprocess, csv
from datetime import datetime
from lpmp.data import load_edges

def run(cmd):
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sle","gat"], required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--node2vec", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--layers", nargs="*", default=None, help="Optional subset of layers; defaults to all in edges.")
    p.add_argument("--direction", choices=["maximize","minimize"], default="maximize")
    p.add_argument("--metric", type=str, default="f1_macro", help="Optuna objective metric (from test json)")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL (sqlite:///study.db) for parallelism")
    p.add_argument("--study-prefix", type=str, default="lpmp_layer")
    # Defaults for non-optimized args that we still pass through
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=None, help="If set, fixes lr; otherwise BO tunes it.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(args.outdir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    summary_csv = os.path.join(args.outdir, f"all_layers_optuna_{args.mode}_{stamp}.csv")

    data = load_edges(args.edges)
    layers = args.layers or data.layers

    with open(summary_csv, "w", newline="") as f:
        fields = ["layer","best_value","run_name","test_loss","acc","f1_macro","auc_approx","best_params"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for L in layers:
            # 1) Per-layer study
            study_name = f"{args.study_prefix}_{args.mode}_{L}"
            optuna_cmd = [
                sys.executable, "scripts/optuna_search.py",
                "--mode", args.mode,
                "--edges", args.edges,
                "--node2vec", args.node2vec,
                "--target-layer", L,
                "--outdir", args.outdir,
                "--n-trials", str(args.n_trials),
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--seed", str(args.seed),
                "--direction", args.direction,
                "--metric", args.metric,
                "--study-name", study_name,
            ]
            if args.storage:
                optuna_cmd += ["--storage", args.storage]
            run(optuna_cmd)

            # Read summary JSON
            # It is named optuna_summary_<mode>_<timestamp>.json; find the latest
            best_summary = None
            latest_path = None
            for fname in sorted(os.listdir(args.outdir)):
                if fname.startswith("optuna_summary_") and fname.endswith(".json"):
                    latest_path = os.path.join(args.outdir, fname)
            if latest_path:
                with open(latest_path) as jf:
                    best_summary = json.load(jf)

            # 2) Re-run CLI on best params for this layer (if we found them)
            run_name = f"{args.mode}_{L}_optuna_best_{stamp}"
            cli_cmd = [
                sys.executable, "-m", "lpmp.cli", args.mode,
                "--edges", args.edges,
                "--node2vec", args.node2vec,
                "--target-layer", L,
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--seed", str(args.seed),
                "--outdir", runs_dir,
                "--run-name", run_name,
                "--batch-size", str(args.batch_size),
            ]

            # If lr not fixed and best params exist, pull lr; else use args.lr or default from CLI
            if best_summary and "best_params" in best_summary and ("lr" in best_summary["best_params"] or args.lr is not None):
                lr = args.lr if args.lr is not None else best_summary["best_params"].get("lr", None)
                if lr is not None:
                    cli_cmd += ["--lr", str(lr)]
            elif args.lr is not None:
                cli_cmd += ["--lr", str(args.lr)]

            # Mode-specific best params
            if best_summary and "best_params" in best_summary:
                bp = best_summary["best_params"]
                if args.mode == "sle":
                    for k in ["sle_nhead","sle_layers","sle_ff","sle_dropout"]:
                        if k in bp:
                            cli_cmd += [f"--{k.replace('_','-')}", str(bp[k])]
                else:
                    for k in ["gat_heads","gat_dropout","tfm_layers","tfm_heads","tfm_ff","tfm_dropout"]:
                        if k in bp:
                            cli_cmd += [f"--{k.replace('_','-')}", str(bp[k])]

            run(cli_cmd)

            # 3) Collect test metrics
            test_json = os.path.join(runs_dir, f"{run_name}_test.json")
            row = {"layer": L, "run_name": run_name, "best_value": None, "best_params": None}
            if best_summary:
                row["best_value"] = best_summary.get("best_value")
                row["best_params"] = json.dumps(best_summary.get("best_params", {}))

            if os.path.exists(test_json):
                with open(test_json) as jf:
                    te = json.load(jf)
                row.update({
                    "test_loss": te.get("test_loss"),
                    "acc": te.get("acc"),
                    "f1_macro": te.get("f1_macro"),
                    "auc_approx": te.get("auc_approx"),
                })
            w.writerow(row)

    print("Wrote all-layers Optuna summary:", summary_csv)

if __name__ == "__main__":
    main()
