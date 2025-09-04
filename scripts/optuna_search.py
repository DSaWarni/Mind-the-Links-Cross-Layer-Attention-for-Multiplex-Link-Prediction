
"""
Optuna-based Bayesian Optimization (TPE by default) for hyperparameter tuning.

Usage (SLE):
    python scripts/optuna_search.py           --mode sle           --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges           --node2vec ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl           --target-layer friendship           --outdir ./results_sle_optuna           --n-trials 40 --epochs 30

Usage (GAT):
    python scripts/optuna_search.py           --mode gat           --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges           --node2vec ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl           --target-layer friendship           --outdir ./results_gat_optuna           --n-trials 40 --epochs 30
"""
import argparse, os, sys, json, subprocess, csv
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
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g., sqlite:///study.db for parallel runs")
    p.add_argument("--study-name", type=str, default=None, help="Name for the Optuna study")
    p.add_argument("--direction", choices=["maximize","minimize"], default="maximize", help="Optimize F1 by default")
    p.add_argument("--metric", type=str, default="f1_macro", help="Metric to optimize from test json (e.g., f1_macro, acc, auc_approx, -test_loss)")
    args = p.parse_args()

    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except Exception as e:
        print("Optuna is required. Install with: pip install optuna", file=sys.stderr)
        raise

    os.makedirs(args.outdir, exist_ok=True)
    runs_dir = os.path.join(args.outdir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction=args.direction,
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(args.storage and args.study_name),
    )

    def objective(trial: "optuna.trial.Trial"):
        # Common hyperparams
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 256, 1024, step=64)

        cmd = [
            sys.executable, "-m", "lpmp.cli", args.mode,
            "--edges", args.edges, "--node2vec", args.node2vec,
            "--target-layer", args.target_layer,
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--seed", str(args.seed),
            "--outdir", runs_dir,
            "--run-name", f"optuna_{args.mode}_{trial.number:04d}",
            "--batch-size", str(batch_size),
            "--lr", str(lr),
        ]

        if args.mode == "sle":
            sle_nhead = trial.suggest_categorical("sle_nhead", [4, 8, 12])
            sle_layers = trial.suggest_int("sle_layers", 2, 4)
            sle_ff = trial.suggest_categorical("sle_ff", [256, 512, 768])
            sle_dropout = trial.suggest_float("sle_dropout", 0.05, 0.2)
            cmd += ["--sle-nhead", str(sle_nhead),
                    "--sle-layers", str(sle_layers),
                    "--sle-ff", str(sle_ff),
                    "--sle-dropout", str(sle_dropout)]
        else:
            gat_heads = trial.suggest_categorical("gat_heads", [4, 8, 12])
            gat_dropout = trial.suggest_float("gat_dropout", 0.05, 0.2)
            tfm_layers = trial.suggest_int("tfm_layers", 2, 4)
            tfm_heads = trial.suggest_categorical("tfm_heads", [4, 8, 12])
            tfm_ff = trial.suggest_categorical("tfm_ff", [256, 512, 768])
            tfm_dropout = trial.suggest_float("tfm_dropout", 0.05, 0.2)
            cmd += ["--gat-heads", str(gat_heads),
                    "--gat-dropout", str(gat_dropout),
                    "--tfm-layers", str(tfm_layers),
                    "--tfm-heads", str(tfm_heads),
                    "--tfm-ff", str(tfm_ff),
                    "--tfm-dropout", str(tfm_dropout)]

        # Run training process
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)

        test_json = os.path.join(runs_dir, f"optuna_{args.mode}_{trial.number:04d}_test.json")
        if not os.path.exists(test_json):
            # penalize missing/failed run
            return 0.0 if args.direction == "maximize" else 1e9

        with open(test_json) as f:
            te = json.load(f)

        # Choose metric
        if args.metric == "-test_loss":
            value = -float(te.get("test_loss", 1e9))
        else:
            value = float(te.get(args.metric, 0.0))

        # report for pruning
        trial.report(value, step=args.epochs)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return value

    print("Starting study:", study.study_name or "<unnamed>")
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # Save study results
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.outdir, f"optuna_summary_{args.mode}_{stamp}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
        }, f, indent=2)

    # Also dump CSV of all trials
    csv_path = os.path.join(args.outdir, f"optuna_trials_{args.mode}_{stamp}.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        fields = ["number","value","state"] + sorted({k for t in study.trials for k in t.params.keys()})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in study.trials:
            row = {"number": t.number, "value": t.value, "state": str(t.state)}
            row.update(t.params)
            w.writerow(row)

    print("Saved:", summary_path)
    print("Saved:", csv_path)

if __name__ == "__main__":
    main()
