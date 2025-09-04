
"""
Train/evaluate one run per target layer and produce a consolidated CSV.
Uses the same CLI (lpmp.cli) under the hood to keep behavior identical.

Example (SLE):
    python scripts/all_layers.py           --mode sle           --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges           --node2vec ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl           --outdir ./results_sle_all           --epochs 30 --batch-size 512 --lr 1e-3 --seed 42

Example (GAT):
    python scripts/all_layers.py           --mode gat           --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges           --node2vec ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl           --outdir ./results_gat_all           --epochs 30 --batch-size 512 --lr 1e-3 --seed 42           --gat-heads 4 --gat-dropout 0.1 --tfm-layers 2 --tfm-heads 4 --tfm-ff 512 --tfm-dropout 0.1
"""
import argparse, os, sys, json, subprocess, csv
from datetime import datetime
from lpmp.data import load_edges

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sle","gat"], required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--node2vec", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-val-test", type=float, nargs=3, default=[0.7,0.15,0.15])
    p.add_argument("--include-target-in-context", action="store_true", default=None,
                   help="If set, override model default for including target layer in context sequence.")
    p.add_argument("--layers", nargs="*", default=None, help="Optional subset of layers to run. Defaults to all found in edges.")
    # SLE params
    p.add_argument("--sle-nhead", type=int, default=4)
    p.add_argument("--sle-layers", type=int, default=2)
    p.add_argument("--sle-ff", type=int, default=512)
    p.add_argument("--sle-dropout", type=float, default=0.1)
    # GAT params
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument("--gat-dropout", type=float, default=0.1)
    p.add_argument("--tfm-layers", type=int, default=2)
    p.add_argument("--tfm-heads", type=int, default=4)
    p.add_argument("--tfm-ff", type=int, default=512)
    p.add_argument("--tfm-dropout", type=float, default=0.1)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(args.outdir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    summary_csv = os.path.join(args.outdir, f"all_layers_{args.mode}_{stamp}.csv")

    # Discover layers
    data = load_edges(args.edges)
    layers = args.layers or data.layers

    # Write consolidated CSV
    with open(summary_csv, "w", newline="") as f:
        fields = ["layer","test_loss","acc","f1_macro","auc_approx","run_name"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for L in layers:
            run_name = f"{args.mode}_{L}_{stamp}"
            cmd = [
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
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--num-workers", str(args.num_workers),
                "--train-val-test", *[str(x) for x in args.train_val_test],
            ]

            # Include/exclude target override if provided
            if args.include_target_in_context is not None:
                if args.include_target_in_context:
                    cmd += ["--include-target-in-context"]
                else:
                    # absence of flag keeps default False for GAT; for SLE we can't pass False flag,
                    # so we rely on default unless user explicitly sets True.
                    pass

            # Model-specific params
            if args.mode == "sle":
                cmd += ["--sle-nhead", str(args.sle_nhead),
                        "--sle-layers", str(args.sle_layers),
                        "--sle-ff", str(args.sle_ff),
                        "--sle-dropout", str(args.sle_dropout)]
            else:
                cmd += ["--gat-heads", str(args.gat_heads),
                        "--gat-dropout", str(args.gat_dropout),
                        "--tfm-layers", str(args.tfm_layers),
                        "--tfm-heads", str(args.tfm_heads),
                        "--tfm-ff", str(args.tfm_ff),
                        "--tfm-dropout", str(args.tfm_dropout)]

            print("Running:", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(proc.stdout)

            test_json = os.path.join(runs_dir, f"{run_name}_test.json")
            row = {"layer": L, "run_name": run_name}
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

    print("Wrote all-layers summary:", summary_csv)

if __name__ == "__main__":
    main()
