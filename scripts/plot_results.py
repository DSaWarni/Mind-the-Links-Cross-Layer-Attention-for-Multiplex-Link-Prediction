
"""
Plot sweep/random-search results CSVs.
Generates simple matplotlib figures (no seaborn; one chart per figure) for:
  - Scatter: F1 vs Loss
  - Scatter: AUC vs F1
  - Metric vs learning rate (log-x)
  - Metric vs batch size
  - Metric vs discrete hyperparameters (bar plots)

Usage:
    python scripts/plot_results.py --csv path/to/sweep_or_random_search.csv --outdir figs/
"""
import argparse, os, csv, math
import matplotlib.pyplot as plt

def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    return rows

def to_float(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "nan")))
        except Exception:
            out.append(float("nan"))
    return out

def uniq(vals):
    seen = set(); out=[]
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--metric", default="test_f1_macro", help="Metric to plot on y-axis for some plots")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = read_rows(args.csv)

    # 1) F1 vs Loss
    f1 = to_float(rows, "test_f1_macro")
    loss = to_float(rows, "test_loss")
    plt.figure()
    plt.scatter(loss, f1)
    plt.xlabel("Test Loss")
    plt.ylabel("Test F1 (macro)")
    plt.title("F1 vs Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "f1_vs_loss.png"))
    plt.close()

    # 2) AUC vs F1
    auc = to_float(rows, "test_auc_approx")
    plt.figure()
    plt.scatter(f1, auc)
    plt.xlabel("Test F1 (macro)")
    plt.ylabel("Test AUC (approx)")
    plt.title("AUC vs F1")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "auc_vs_f1.png"))
    plt.close()

    # Learning-rate and batch size may be named differently depending on CSV
    lr = to_float(rows, "lr")
    bs = to_float(rows, "batch_size")

    # 3) Metric vs learning rate (log-x)
    y = to_float(rows, args.metric)
    plt.figure()
    # Filter valid points
    xy = [(lx, yy) for lx, yy in zip(lr, y) if not (math.isnan(lx) or math.isnan(yy) or lx<=0)]
    if xy:
        xs, ys = zip(*xy)
        plt.scatter(xs, ys)
        plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} vs lr")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.metric}_vs_lr.png"))
    plt.close()

    # 4) Metric vs batch size
    plt.figure()
    xy = [(bx, yy) for bx, yy in zip(bs, y) if not (math.isnan(bx) or math.isnan(yy))]
    if xy:
        xs, ys = zip(*xy)
        plt.scatter(xs, ys)
    plt.xlabel("Batch size")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} vs batch size")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.metric}_vs_batch.png"))
    plt.close()

    # 5) Bar plots for discrete hparams present in CSV
    discrete_keys = ["sle_nhead","sle_layers","sle_ff","sle_dropout",
                     "gat_heads","gat_dropout","tfm_layers","tfm_heads","tfm_ff","tfm_dropout"]
    for k in discrete_keys:
        vals = [r.get(k) for r in rows if r.get(k) not in (None, "", "None")]
        if not vals:
            continue
        # average metric per value
        buckets = {}
        for r in rows:
            v = r.get(k)
            if v in (None, "", "None"): continue
            try:
                m = float(r.get(args.metric, "nan"))
            except Exception:
                continue
            if m!=m: # NaN
                continue
            buckets.setdefault(v, []).append(m)
        if not buckets:
            continue
        xs = sorted(buckets.keys(), key=lambda x: float(x) if x.replace('.','',1).isdigit() else x)
        ys = [sum(buckets[x])/len(buckets[x]) for x in xs]
        plt.figure()
        plt.bar(range(len(xs)), ys)
        plt.xticks(range(len(xs)), xs, rotation=45, ha="right")
        plt.ylabel(args.metric)
        plt.title(f"{args.metric} vs {k}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"{args.metric}_vs_{k}.png"))
        plt.close()

    print("Wrote figures to", args.outdir)

if __name__ == "__main__":
    main()
