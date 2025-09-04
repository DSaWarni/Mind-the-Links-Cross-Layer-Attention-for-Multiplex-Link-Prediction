
"""
Simple parameter sensitivity / grid search runner.
Launches multiple CLI runs and aggregates test metrics into a CSV.
Usage:
    python scripts/sweep.py --mode sle --edges ... --node2vec ... --target-layer friendship --outdir results/
"""
import argparse, itertools, os, json, subprocess, sys, csv, shutil
from datetime import datetime

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sle","gat"], required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--node2vec", required=True)
    p.add_argument("--target-layer", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    # Define search spaces (edit freely)
    p.add_argument("--grid", type=str, default=None,
                   help="JSON dict of param lists. Example: '{"lr":[1e-3,5e-4],"batch_size":[256,512]}'")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(args.outdir, f"sweep_{args.mode}_{stamp}.csv")

    # Default grids (safe, small)
    if args.grid:
        grid = json.loads(args.grid)
    else:
        if args.mode == "sle":
            grid = {
                "lr": [1e-3, 5e-4],
                "batch_size": [256, 512],
                "sle_nhead": [4, 8],
                "sle_layers": [2, 3],
                "sle_ff": [256, 512],
                "sle_dropout": [0.1, 0.2],
            }
        else:
            grid = {
                "lr": [1e-3, 5e-4],
                "batch_size": [256, 512],
                "gat_heads": [4, 8],
                "gat_dropout": [0.1, 0.2],
                "tfm_layers": [2, 3],
                "tfm_heads": [4, 8],
                "tfm_ff": [256, 512],
                "tfm_dropout": [0.1, 0.2],
            }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    with open(summary_csv, "w", newline="") as f:
        fieldnames = ["mode","target_layer","seed","epochs"] + keys + ["test_loss","test_acc","test_f1_macro","test_auc_approx","run_dir"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for combo in combos:
            params = dict(zip(keys, combo))
            run_dir = os.path.join(args.outdir, "runs")
            os.makedirs(run_dir, exist_ok=True)
            run_name = f"{args.mode}_{args.target_layer}_" + "_".join(f"{k}-{v}" for k,v in params.items())

            cmd = [
                sys.executable, "-m", "lpmp.cli", args.mode,
                "--edges", args.edges,
                "--node2vec", args.node2vec,
                "--target-layer", args.target_layer,
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--seed", str(args.seed),
                "--outdir", run_dir,
                "--run-name", run_name,
            ]

            # add learning-rate / batch etc if present
            if "lr" in params: cmd += ["--lr", str(params["lr"])]
            if "batch_size" in params: cmd += ["--batch-size", str(params["batch_size"])]

            # mode-specific
            if args.mode == "sle":
                for k in ["sle_nhead","sle_layers","sle_ff","sle_dropout"]:
                    if k in params:
                        cmd += [f"--{k.replace('_','-')}", str(params[k])]
            else:
                for k in ["gat_heads","gat_dropout","tfm_layers","tfm_heads","tfm_ff","tfm_dropout"]:
                    if k in params:
                        cmd += [f"--{k.replace('_','-')}", str(params[k])]

            print("Running:", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(proc.stdout)

            test_json = os.path.join(run_dir, f"{run_name}_test.json")
            if os.path.exists(test_json):
                with open(test_json) as jf:
                    te = json.load(jf)
                row = {
                    "mode": args.mode, "target_layer": args.target_layer, "seed": args.seed, "epochs": args.epochs,
                    **params,
                    "test_loss": te.get("test_loss"),
                    "test_acc": te.get("acc"),
                    "test_f1_macro": te.get("f1_macro"),
                    "test_auc_approx": te.get("auc_approx"),
                    "run_dir": run_dir
                }
                w.writerow(row)
            else:
                row = {"mode": args.mode, "target_layer": args.target_layer, **params, "run_dir": run_dir}
                w.writerow(row)

    print("Wrote summary:", summary_csv)

if __name__ == "__main__":
    main()
