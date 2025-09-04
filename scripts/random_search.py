
"""
Random search for hyperparameter sensitivity.
Usage (SLE):
    python scripts/random_search.py --mode sle --trials 20 --edges ... --node2vec ... --target-layer friendship --outdir results_sle_rs

Usage (GAT):
    python scripts/random_search.py --mode gat --trials 20 --edges ... --node2vec ... --target-layer friendship --outdir results_gat_rs
"""
import argparse, os, sys, json, random, subprocess, csv
from datetime import datetime

def float_range(lo, hi):
    return lo + random.random()*(hi-lo)

def int_choice(choices):
    return random.choice(choices)

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
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--batch-min", type=int, default=256)
    p.add_argument("--batch-max", type=int, default=1024)
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--lr-max", type=float, default=2e-3)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(args.outdir, f"random_search_{args.mode}_{stamp}.csv")
    runs_dir = os.path.join(args.outdir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    if args.mode == "sle":
        discrete = {
            "sle_nhead": [4, 8, 12],
            "sle_layers": [2, 3, 4],
            "sle_ff": [256, 512, 768],
            "sle_dropout": [0.05, 0.1, 0.2],
        }
    else:
        discrete = {
            "gat_heads": [4, 8, 12],
            "gat_dropout": [0.05, 0.1, 0.2],
            "tfm_layers": [2, 3, 4],
            "tfm_heads": [4, 8, 12],
            "tfm_ff": [256, 512, 768],
            "tfm_dropout": [0.05, 0.1, 0.2],
        }

    with open(summary_csv, "w", newline="") as f:
        fields = ["trial","mode","target_layer","seed","epochs","batch_size","lr"] + list(discrete.keys()) + ["test_loss","test_acc","test_f1_macro","test_auc_approx","run_name"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for t in range(1, args.trials+1):
            batch_size = random.randrange(args.batch_min//64, args.batch_max//64 + 1) * 64
            lr = float_range(args.lr_min, args.lr_max)

            params = {}
            for k, vals in discrete.items():
                params[k] = int_choice(vals)

            run_name = f"rs_{args.mode}_{args.target_layer}_t{t:03d}"

            cmd = [
                sys.executable, "-m", "lpmp.cli", args.mode,
                "--edges", args.edges, "--node2vec", args.node2vec,
                "--target-layer", args.target_layer,
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--seed", str(args.seed),
                "--outdir", runs_dir,
                "--run-name", run_name,
                "--batch-size", str(batch_size),
                "--lr", str(lr),
            ]

            if args.mode == "sle":
                for k in ["sle_nhead","sle_layers","sle_ff","sle_dropout"]:
                    cmd += [f"--{k.replace('_','-')}", str(params[k])]
            else:
                for k in ["gat_heads","gat_dropout","tfm_layers","tfm_heads","tfm_ff","tfm_dropout"]:
                    cmd += [f"--{k.replace('_','-')}", str(params[k])]

            print("Running:", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(proc.stdout)

            test_json = os.path.join(runs_dir, f"{run_name}_test.json")
            row = {"trial": t, "mode": args.mode, "target_layer": args.target_layer, "seed": args.seed, "epochs": args.epochs,
                   "batch_size": batch_size, "lr": lr, "run_name": run_name}
            row.update(params)
            if os.path.exists(test_json):
                with open(test_json) as jf:
                    te = json.load(jf)
                row["test_loss"] = te.get("test_loss")
                row["test_acc"] = te.get("acc")
                row["test_f1_macro"] = te.get("f1_macro")
                row["test_auc_approx"] = te.get("auc_approx")
            w.writerow(row)

    print("Wrote random search summary:", summary_csv)
if __name__ == "__main__":
    main()
