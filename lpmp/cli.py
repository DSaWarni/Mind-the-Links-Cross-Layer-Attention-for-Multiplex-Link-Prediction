import argparse, os, json
import numpy as np
import torch
from datetime import datetime

from lpmp.data import (
    load_edges, build_union_pairs, labels_for_layer, stratified_split,
    PairDataset, compute_pos_weight, mask_pairs_from_layer
)
from lpmp.train import Trainer
from lpmp.baselines.mell import train_mell, mell_scores_for_layer, mell_prepare, mell_train_one_epoch, mell_scores_for_layer
from lpmp.baselines.hoplp_mul import hoplp_scores, scores_for_pairs
from lpmp.baselines.rmne import train_rmne, rmne_scores, rmne_prepare, rmne_train_one_epoch
import csv


def _safe_args(ns):
    out = {}
    for k, v in vars(ns).items():
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = v
        else:
            out[k] = str(v)  # fallback: stringify Paths etc.
    return out

def _scores_from_binary_logits(logits: torch.Tensor,
                               targets: torch.Tensor,
                               threshold: float = 0.5) -> dict:
    """
    logits: [B,1] (raw scores); targets: [B,1] in {0,1}
    Returns: acc, f1_macro, auc_approx  (same definitions as Trainer)
    """
    if logits.ndim == 1:
        logits = logits.view(-1, 1)
    if targets.ndim == 1:
        targets = targets.view(-1, 1)

    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y = targets.detach().cpu().numpy().reshape(-1).astype(int)
    yhat = (probs >= float(threshold)).astype(int)

    # acc
    acc = float((yhat == y).mean())

    # macro-F1
    eps = 1e-9
    f1s = []
    for cls in (0, 1):
        tp = np.sum((y == cls) & (yhat == cls))
        fp = np.sum((y != cls) & (yhat == cls))
        fn = np.sum((y == cls) & (yhat != cls))
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    f1_macro = float(np.mean(f1s))

    # approx AUC (WMW)
    pos = probs[y == 1]; neg = probs[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = 0.5
    else:
        all_scores = np.concatenate([pos, neg])
        order = np.argsort(all_scores, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(all_scores)) + 1
        sum_pos = ranks[: len(pos)].sum()
        auc = (sum_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
        auc = float(auc)

    return {"acc": acc, "f1_macro": f1_macro, "auc_approx": auc}


def _col(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is shaped [B,1] and float32 for BCEWithLogitsLoss."""
    if x.ndim == 1:
        x = x.view(-1, 1)
    return x.to(torch.float32)




# ------------------------------ utils ------------------------------

def load_node2vec(pkl_path: str):
    import pickle
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    out = {}
    for L, arr in obj.items():
        out[str(L)] = torch.as_tensor(arr, dtype=torch.float32)
    return out

def add_common_shared(sp):
    # Dataset/schema
    sp.add_argument("--edges", type=str, required=True, help="Path to .edges file")
    sp.add_argument("--layer-col", type=str, default=None)
    sp.add_argument("--start-col", type=str, default=None)
    sp.add_argument("--end-col", type=str, default=None)
    sp.add_argument("--id-base", type=int, default=1, choices=[0,1], help="0 if node ids are 0-based; else 1")
    sp.add_argument("--has-header", action="store_true")

    # Task & fairness protocol
    sp.add_argument("--target-layer", type=str, required=True, help="Target layer to predict")
    sp.add_argument("--train-val-test", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TR","VA","TE"))
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    sp.add_argument("--outdir", type=str, default=None)

    # # Embeddings: precomputed OR on-the-fly
    # group = sp.add_mutually_exclusive_group(required=True)
    # group.add_argument("--node2vec", type=str, help="Path to precomputed layer embeddings .pkl")
    # group.add_argument("--embed-backend", type=str, choices=["node2vec","corewalk","random"], help="Compute embeddings on the fly")
    # sp.add_argument("--emb-dim", type=int, default=128)

    # # Node2Vec params
    # sp.add_argument("--n2v-p", type=float, default=1.0)
    # sp.add_argument("--n2v-q", type=float, default=1.0)
    # sp.add_argument("--n2v-walk-len", type=int, default=50)
    # sp.add_argument("--n2v-walks-per-node", type=int, default=10)

    # # Corewalk params
    # sp.add_argument("--core-temp", type=float, default=1.0)
    # sp.add_argument("--core-walk-len", type=int, default=40)
    # sp.add_argument("--core-walks-per-node", type=int, default=10)

    # # Skip-gram params
    # sp.add_argument("--sgns-window", type=int, default=10)
    # sp.add_argument("--sgns-epochs", type=int, default=1)
    # sp.add_argument("--sgns-batch", type=int, default=8192)
    # sp.add_argument("--sgns-lr", type=float, default=1e-3)
    # sp.add_argument("--sgns-neg-k", type=int, default=1)

    # sp.add_argument("--save-embeddings", type=str, default=None, help="If set, save computed embeddings as .pkl")

    # Frozen splits + masking (apples-to-apples)
    sp.add_argument("--splits", type=str, default=None,
                    help="Path to JSON with frozen indices {train:[], val:[], test:[]}")
    sp.add_argument("--mask-test-in-target", action="store_true",
                    help="Remove TARGET-LAYER test edges from structure before unsupervised steps")

    # Optim/training
    sp.add_argument("--epochs", type=int, default=50)
    sp.add_argument("--batch-size", type=int, default=512)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--weight-decay", type=float, default=0.0)
    sp.add_argument("--num-workers", type=int, default=4)
    sp.add_argument("--grad-clip", type=float, default=1.0)
    sp.add_argument("--log-every", type=int, default=1)
    sp.add_argument("--run-name", type=str, default=None)

    sp.add_argument("--early-stopping", type=int, default=20,
               help="Patience for early stopping on validation F1 (0=disabled).")
    sp.add_argument("--csv-log", action="store_true",
               help="If set, write a training_log.csv in the outdir.")




def add_embedding_args(sp):
    # Embeddings: precomputed OR on-the-fly (ONLY for SLE/GAT)
    group = sp.add_mutually_exclusive_group(required=True)
    group.add_argument("--node2vec", type=str, help="Path to precomputed layer embeddings .pkl")
    group.add_argument("--embed-backend", type=str, choices=["node2vec","corewalk","random"],
                       help="Compute embeddings on the fly")
    sp.add_argument("--emb-dim", type=int, default=128)

    # Node2Vec params
    sp.add_argument("--n2v-p", type=float, default=1.0)
    sp.add_argument("--n2v-q", type=float, default=1.0)
    sp.add_argument("--n2v-walk-len", type=int, default=50)
    sp.add_argument("--n2v-walks-per-node", type=int, default=10)

    # Corewalk params
    sp.add_argument("--core-temp", type=float, default=1.0)
    sp.add_argument("--core-walk-len", type=int, default=40)
    sp.add_argument("--core-walks-per-node", type=int, default=10)

    # Skip-gram params
    sp.add_argument("--sgns-window", type=int, default=10)
    sp.add_argument("--sgns-epochs", type=int, default=1)
    sp.add_argument("--sgns-batch", type=int, default=8192)
    sp.add_argument("--sgns-lr", type=float, default=1e-3)
    sp.add_argument("--sgns-neg-k", type=int, default=1)

    sp.add_argument("--save-embeddings", type=str, default=None,
                    help="If set, save computed embeddings as .pkl")



# ------------------------------ runners ------------------------------

def prepare_data_and_embeddings(args):
    # Load graph
    data = load_edges(
        args.edges,
        layer_col=args.layer_col, start_col=args.start_col, end_col=args.end_col,
        id_base=args.id_base, has_header=args.has_header
    )
    union_pairs = sorted(list(build_union_pairs(data.edge_sets)))

    # Splits (frozen or fresh)
    splits = None
    if args.splits:
        with open(args.splits) as f:
            splits = json.load(f)

    # Mask target-layer TEST edges for unsupervised steps (if requested)
    edge_sets_unsup = data.edge_sets
    if args.mask_test_in_target and splits is not None:
        test_pairs = [union_pairs[i] for i in splits["test"]]
        edge_sets_unsup = mask_pairs_from_layer(data.edge_sets, test_pairs, args.target_layer)

    # Build or load embeddings
    if args.node2vec is not None:
        node2vec = load_node2vec(args.node2vec)
    else:
        from lpmp.embeddings import build_layer_embeddings
        node2vec = build_layer_embeddings(
            num_nodes=data.num_nodes,
            edge_sets=edge_sets_unsup,
            backend=args.embed_backend,
            dim=args.emb_dim,
            n2v_p=args.n2v_p, n2v_q=args.n2v_q,
            n2v_walk_len=args.n2v_walk_len, n2v_walks_per_node=args.n2v_walks_per_node,
            core_temp=args.core_temp, core_walk_len=args.core_walk_len, core_walks_per_node=args.core_walks_per_node,
            sgns_window=args.sgns_window, sgns_epochs=args.sgns_epochs, sgns_batch=args.sgns_batch,
            sgns_lr=args.sgns_lr, sgns_neg_k=args.sgns_neg_k,
            seed=args.seed, device=args.device
        )
        if args.save_embeddings:
            os.makedirs(os.path.dirname(args.save_embeddings), exist_ok=True)
            with open(args.save_embeddings, "wb") as f:
                import pickle
                pickle.dump({k: v.cpu().numpy() for k,v in node2vec.items()}, f)
            print(f"[embeddings] saved to {args.save_embeddings}")

    # Labels and splits for supervised training/eval (ALWAYS on original data.edge_sets)
    y = labels_for_layer(args.target_layer, union_pairs, data.edge_sets)
    idx_all = np.arange(len(union_pairs))
    if splits:
        tr = np.array(splits["train"], dtype=int)
        va = np.array(splits["val"], dtype=int)
        te = np.array(splits["test"], dtype=int)
    else:
        tr, va, te = stratified_split(idx_all, y, *args.train_val_test, seed=args.seed)

    ds_tr = PairDataset([union_pairs[i] for i in tr], y[tr])
    ds_va = PairDataset([union_pairs[i] for i in va], y[va])
    ds_te = PairDataset([union_pairs[i] for i in te], y[te])
    posw = compute_pos_weight(y[tr])

    return data, node2vec, ds_tr, ds_va, ds_te, posw

def run_sle(args):
    data, node2vec, ds_tr, ds_va, ds_te, posw = prepare_data_and_embeddings(args)
    from lpmp.models.trans_sle import SLETransformer
    d_node = next(iter(node2vec.values())).shape[1]
    model = SLETransformer(
        data.layers, node2vec,
        d_model=2*d_node, nhead=args.sle_nhead, num_layers=args.sle_layers,
        dim_feedforward=args.sle_ff, dropout=args.sle_dropout,
        include_target_in_context=args.include_target_in_context
    )
    trainer = Trainer(model, device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                      num_workers=args.num_workers, grad_clip=args.grad_clip)
    history, test_scores = trainer.fit(
        target_layer=args.target_layer,
        train_set=ds_tr, val_set=ds_va, test_set=ds_te,
        batch_size=args.batch_size, pos_weight_train=posw, pos_weight_eval=posw,
        epochs=args.epochs, log_every=args.log_every,
        csv_path = os.path.join(args.outdir, "training_log.csv") if args.outdir else None,
        early_stopping_patience= args.early_stopping
    )
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = args.run_name or f"sle_{args.target_layer}_{stamp}"
        
        args_json = _safe_args(args)
        with open(os.path.join(args.outdir, f"{run}_args.json"), "w") as f:
            json.dump(args_json, f, indent=2)
        # optional: save history CSV if you want
        with open(os.path.join(args.outdir, f"{run}_test.json"), "w") as f: json.dump(test_scores, f, indent=2)
    return test_scores

def run_gat(args):
    data, node2vec, ds_tr, ds_va, ds_te, posw = prepare_data_and_embeddings(args)
    from lpmp.models.trans_gat import TransGAT
    d_node = next(iter(node2vec.values())).shape[1]
    model = TransGAT(
        data.layers, node2vec, data.edge_index,
        d_node=d_node, d_model=2*d_node,
        gat_heads=args.gat_heads, gat_dropout=args.gat_dropout,
        tfm_layers=args.tfm_layers, tfm_heads=args.tfm_heads,
        tfm_ff=args.tfm_ff, tfm_dropout=args.tfm_dropout,
        include_target_in_context=args.include_target_in_context
    )
    trainer = Trainer(model, device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                      num_workers=args.num_workers, grad_clip=args.grad_clip)
    history, test_scores = trainer.fit(
        target_layer=args.target_layer,
        train_set=ds_tr, val_set=ds_va, test_set=ds_te,
        batch_size=args.batch_size, pos_weight_train=posw, pos_weight_eval=posw,
        epochs=args.epochs, log_every=args.log_every,
        csv_path = os.path.join(args.outdir, "training_log.csv") if args.outdir else None,
        early_stopping_patience= args.early_stopping
    )
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = args.run_name or f"gat_{args.target_layer}_{stamp}"
        
        args_json = _safe_args(args)
        with open(os.path.join(args.outdir, f"{run}_args.json"), "w") as f:
            json.dump(args_json, f, indent=2)
        with open(os.path.join(args.outdir, f"{run}_test.json"), "w") as f: json.dump(test_scores, f, indent=2)
    return test_scores

def run_mell(args):
    # Load & splits
    data = load_edges(args.edges, layer_col=args.layer_col, start_col=args.start_col,
                      end_col=args.end_col, id_base=args.id_base, has_header=args.has_header)
    union_pairs = sorted(list(build_union_pairs(data.edge_sets)))
    if args.splits:
        with open(args.splits) as f: sp = json.load(f)
        tr = np.array(sp["train"], dtype=int); va = np.array(sp["val"], dtype=int); te = np.array(sp["test"], dtype=int)
    else:
        y_tmp = labels_for_layer(args.target_layer, union_pairs, data.edge_sets)
        idx = np.arange(len(union_pairs))
        tr, va, te = stratified_split(idx, y_tmp, *args.train_val_test, seed=args.seed)

    # Leakage mask if requested
    edge_sets_unsup = data.edge_sets
    if args.mask_test_in_target:
        test_pairs = [union_pairs[i] for i in te]
        edge_sets_unsup = mask_pairs_from_layer(data.edge_sets, test_pairs, args.target_layer)

    # Prepare model/opt/dataset for epoch-wise training
    model, opt, neg_sampler, all_pos = mell_prepare(
        edge_sets=edge_sets_unsup, num_nodes=data.num_nodes, layers=data.layers,
        dim=args.dim, lr=args.lr, device=args.device
    )

    y = labels_for_layer(args.target_layer, union_pairs, data.edge_sets)
    tr_pairs = [union_pairs[i] for i in tr]; va_pairs = [union_pairs[i] for i in va]; te_pairs = [union_pairs[i] for i in te]
    tr_y = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(-1)
    va_y = torch.tensor(y[va], dtype=torch.float32).unsqueeze(-1)
    te_y = torch.tensor(y[te], dtype=torch.float32).unsqueeze(-1)
    bce = torch.nn.BCEWithLogitsLoss()

    # CSV log (per-epoch)
    csv_fh = None; csv_writer = None
    if args.csv_log and args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        csv_fh = open(os.path.join(args.outdir, "training_log.csv"), "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["epoch","train_loss","train_f1","train_acc","train_auc","val_loss","val_f1","val_acc","val_auc","best_f1","best_thr"])
        csv_fh.flush()

    patience = max(0, int(args.early_stopping)); patience_left = patience
    best_val_f1 = -1.0; best_thr = 0.5; best_state = None

    for ep in range(1, args.epochs + 1):
        # One epoch of MeLL
        train_loss_epoch = mell_train_one_epoch(model, opt, neg_sampler, all_pos, batch_size=args.batch_size, device=args.device)
        print(f"[MELL] epoch={ep} loss={train_loss_epoch:.4f}")

        # Eval train/val
        tr_logits = mell_scores_for_layer(model, args.target_layer, tr_pairs, device=args.device)
        va_logits = mell_scores_for_layer(model, args.target_layer, va_pairs, device=args.device)
        tr_loss = bce(_col(tr_logits), tr_y).item(); va_loss = bce(_col(va_logits), va_y).item()

        # Pick threshold on validation
        thr_grid = np.linspace(0.05, 0.95, 19)
        cand_thr, cand_f1 = best_thr, -1.0
        for thr in thr_grid:
            s = _scores_from_binary_logits(va_logits, va_y, threshold=float(thr))
            if s["f1_macro"] > cand_f1:
                cand_f1, cand_thr = s["f1_macro"], float(thr)

        tr_scores = _scores_from_binary_logits(tr_logits, tr_y, threshold=cand_thr)
        va_scores = _scores_from_binary_logits(va_logits, va_y, threshold=cand_thr)

        improved = cand_f1 > best_val_f1
        if improved:
            best_val_f1, best_thr = cand_f1, cand_thr
            patience_left = patience
            if hasattr(model, "state_dict"):
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            if patience > 0:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[early stopping] no val F1 improvement for {patience} epochs.")
                    break

        if csv_writer is not None:
            csv_writer.writerow([ep, tr_loss, tr_scores["f1_macro"], tr_scores["acc"], tr_scores["auc_approx"],
                                 va_loss, va_scores["f1_macro"], va_scores["acc"], va_scores["auc_approx"],
                                 best_val_f1, best_thr])
            csv_fh.flush()

    # Restore best before test
    if best_state is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(best_state)

    # Final test with frozen threshold
    te_logits = mell_scores_for_layer(model, args.target_layer, te_pairs, device=args.device)
    te_loss = bce(_col(te_logits), te_y).item()
    te_scores = _scores_from_binary_logits(te_logits, te_y, threshold=best_thr)

    if csv_fh is not None:
        csv_writer.writerow(["TEST","","","","","","","","",best_val_f1,best_thr])
        csv_fh.close()

    out = {"threshold": best_thr,
           "train_loss": tr_loss, "val_loss": va_loss, "test_loss": te_loss,
           "train_f1_macro": tr_scores["f1_macro"], "train_acc": tr_scores["acc"], "train_auc_approx": tr_scores["auc_approx"],
           "val_f1_macro": va_scores["f1_macro"], "val_acc": va_scores["acc"], "val_auc_approx": va_scores["auc_approx"],
           **te_scores}

    if args.outdir:
        with open(os.path.join(args.outdir, f"mell_{args.target_layer}_args.json"), "w") as f:
            json.dump(_safe_args(args), f, indent=2)
        with open(os.path.join(args.outdir, f"mell_{args.target_layer}_test.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out


def run_hoplp(args):
    data = load_edges(args.edges, layer_col=args.layer_col, start_col=args.start_col,
                      end_col=args.end_col, id_base=args.id_base, has_header=args.has_header)
    union_pairs = sorted(list(build_union_pairs(data.edge_sets)))
    if args.splits:
        with open(args.splits) as f: sp = json.load(f)
        tr = np.array(sp["train"], dtype=int); va = np.array(sp["val"], dtype=int); te = np.array(sp["test"], dtype=int)
    else:
        y_tmp = labels_for_layer(args.target_layer, union_pairs, data.edge_sets)
        idx = np.arange(len(union_pairs))
        tr, va, te = stratified_split(idx, y_tmp, *args.train_val_test, seed=args.seed)

    edge_sets_unsup = data.edge_sets
    if args.mask_test_in_target:     # and args.splits:
        test_pairs = [union_pairs[i] for i in te]
        edge_sets_unsup = mask_pairs_from_layer(data.edge_sets, test_pairs, args.target_layer)

    S = hoplp_scores(edge_sets_unsup, data.num_nodes, K=args.K, alpha=args.alpha, weight_mode=args.weight_mode)
    y = labels_for_layer(args.target_layer, union_pairs, data.edge_sets)

    tr_pairs = [union_pairs[i] for i in tr]
    va_pairs = [union_pairs[i] for i in va]
    te_pairs = [union_pairs[i] for i in te]

    tr_logits = scores_for_pairs(S, tr_pairs)
    va_logits = scores_for_pairs(S, va_pairs)
    te_logits = scores_for_pairs(S, te_pairs)

    tr_y = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(-1)
    va_y = torch.tensor(y[va], dtype=torch.float32).unsqueeze(-1)
    te_y = torch.tensor(y[te], dtype=torch.float32).unsqueeze(-1)

    thr_grid = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thr_grid:
        s = _scores_from_binary_logits(va_logits, va_y, threshold=float(thr))
        if s["f1_macro"] > best_f1:
            best_f1, best_thr = s["f1_macro"], float(thr)

    bce = torch.nn.BCEWithLogitsLoss()
    tr_loss = bce(_col(tr_logits), tr_y).item()
    va_loss = bce(_col(va_logits), va_y).item()
    te_loss = bce(_col(te_logits), te_y).item()

    tr_scores = _scores_from_binary_logits(tr_logits, tr_y, threshold=best_thr)
    va_scores = _scores_from_binary_logits(va_logits, va_y, threshold=best_thr)
    te_scores = _scores_from_binary_logits(te_logits, te_y, threshold=best_thr)

    out = {
        "threshold": best_thr,
        "train_loss": tr_loss, **{f"train_{k}": v for k, v in tr_scores.items()},
        "val_loss": va_loss,   **{f"val_{k}":   v for k, v in va_scores.items()},
        "test_loss": te_loss,  **te_scores
    }



    if args.csv_log and args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "training_log.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch","split","loss","f1_macro","acc","auc_approx","threshold"])
            # we don’t have per-epoch logs for these baselines, so just dump final summary
            w.writerow(["-", "train", out["train_loss"], out["train_f1_macro"], out["train_acc"], out["train_auc_approx"], out["threshold"]])
            w.writerow(["-", "val",   out["val_loss"],   out["val_f1_macro"],   out["val_acc"],   out["val_auc_approx"],   out["threshold"]])
            w.writerow(["-", "test",  out["test_loss"],  out["f1_macro"],       out["acc"],       out["auc_approx"],       out["threshold"]])

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

        args_json = _safe_args(args)
        with open(os.path.join(args.outdir, f"hoplp_{args.target_layer}_args.json"), "w") as f:
            json.dump(args_json, f, indent=2)
        with open(os.path.join(args.outdir, f"hoplp_{args.target_layer}_test.json"), "w") as f:
            json.dump(out, f, indent=2)

    return out

def run_rmne(args):
    data = load_edges(args.edges, layer_col=args.layer_col, start_col=args.start_col,
                      end_col=args.end_col, id_base=args.id_base, has_header=args.has_header)
    union_pairs = sorted(list(build_union_pairs(data.edge_sets)))
    if args.splits:
        with open(args.splits) as f: sp = json.load(f)
        tr = np.array(sp["train"], dtype=int); va = np.array(sp["val"], dtype=int); te = np.array(sp["test"], dtype=int)
    else:
        y_tmp = labels_for_layer(args.target_layer, union_pairs, data.edge_sets)
        idx = np.arange(len(union_pairs))
        tr, va, te = stratified_split(idx, y_tmp, *args.train_val_test, seed=args.seed)

    edge_sets_unsup = data.edge_sets
    if args.mask_test_in_target:
        test_pairs = [union_pairs[i] for i in te]
        edge_sets_unsup = mask_pairs_from_layer(data.edge_sets, test_pairs, args.target_layer)

    # Prepare model/opt/dataset for epoch-wise training
    rm_model, rm_opt, pairs, rng = rmne_prepare(
        num_nodes=data.num_nodes, edge_sets=edge_sets_unsup, dim=args.dim,
        K=args.K, Le=args.Le, window=args.window, lr=args.lr, r=args.r, t=args.t,
        seed=args.seed, device=args.device
    )

    y = labels_for_layer(args.target_layer, union_pairs, data.edge_sets)
    tr_pairs = [union_pairs[i] for i in tr]; va_pairs = [union_pairs[i] for i in va]; te_pairs = [union_pairs[i] for i in te]
    tr_y = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(-1)
    va_y = torch.tensor(y[va], dtype=torch.float32).unsqueeze(-1)
    te_y = torch.tensor(y[te], dtype=torch.float32).unsqueeze(-1)
    bce = torch.nn.BCEWithLogitsLoss()

    # CSV log
    csv_fh = None; csv_writer = None
    if args.csv_log and args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        csv_fh = open(os.path.join(args.outdir, "training_log.csv"), "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["epoch","train_loss","train_f1","train_acc","train_auc","val_loss","val_f1","val_acc","val_auc","best_f1","best_thr"])
        csv_fh.flush()

    patience = max(0, int(args.early_stopping)); patience_left = patience
    best_val_f1 = -1.0; best_thr = 0.5; best_state = None

    for ep in range(1, args.epochs + 1):
        # One epoch of RMNE
        train_loss_epoch = rmne_train_one_epoch(rm_model, rm_opt, pairs, data.num_nodes, rng,
                                                batch_size=args.batch_size, device=args.device)
        print(f"[RMNE] epoch={ep} loss={train_loss_epoch:.4f}")

        # Current embeddings for scoring
        cur_emb = rm_model.in_emb.weight.data.detach().cpu()

        # Eval train/val
        tr_logits = rmne_scores(cur_emb, tr_pairs)
        va_logits = rmne_scores(cur_emb, va_pairs)
        tr_loss = bce(_col(tr_logits), tr_y).item(); va_loss = bce(_col(va_logits), va_y).item()

        # Pick threshold on validation
        thr_grid = np.linspace(0.05, 0.95, 19)
        cand_thr, cand_f1 = best_thr, -1.0
        for thr in thr_grid:
            s = _scores_from_binary_logits(va_logits, va_y, threshold=float(thr))
            if s["f1_macro"] > cand_f1:
                cand_f1, cand_thr = s["f1_macro"], float(thr)

        tr_scores = _scores_from_binary_logits(tr_logits, tr_y, threshold=cand_thr)
        va_scores = _scores_from_binary_logits(va_logits, va_y, threshold=cand_thr)

        improved = cand_f1 > best_val_f1
        if improved:
            best_val_f1, best_thr = cand_f1, cand_thr
            patience_left = patience
            # snapshot best state
            best_state = {k: v.detach().cpu() for k, v in rm_model.state_dict().items()}
        else:
            if patience > 0:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[early stopping] no val F1 improvement for {patience} epochs.")
                    break

        if csv_writer is not None:
            csv_writer.writerow([ep, tr_loss, tr_scores["f1_macro"], tr_scores["acc"], tr_scores["auc_approx"],
                                 va_loss, va_scores["f1_macro"], va_scores["acc"], va_scores["auc_approx"],
                                 best_val_f1, best_thr])
            csv_fh.flush()

    # Restore best before test
    if best_state is not None:
        rm_model.load_state_dict({k: v.to(rm_model.state_dict()[k].device) for k, v in best_state.items()})
    best_emb = rm_model.in_emb.weight.data.detach().cpu()

    # Final test with frozen threshold
    te_logits = rmne_scores(best_emb, te_pairs)
    te_loss = bce(_col(te_logits), te_y).item()
    te_scores = _scores_from_binary_logits(te_logits, te_y, threshold=best_thr)

    if csv_fh is not None:
        csv_writer.writerow(["TEST","","","","","","","","",best_val_f1,best_thr])
        csv_fh.close()

    out = {"threshold": best_thr,
           "train_loss": tr_loss, "val_loss": va_loss, "test_loss": te_loss,
           "train_f1_macro": tr_scores["f1_macro"], "train_acc": tr_scores["acc"], "train_auc_approx": tr_scores["auc_approx"],
           "val_f1_macro": va_scores["f1_macro"], "val_acc": va_scores["acc"], "val_auc_approx": va_scores["auc_approx"],
           **te_scores}

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        args_json = _safe_args(args)
        with open(os.path.join(args.outdir, f"rmne_{args.target_layer}_args.json"), "w") as f:
            json.dump(args_json, f, indent=2)
        with open(os.path.join(args.outdir, f"rmne_{args.target_layer}_test.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out




def run_freeze(args):
    data = load_edges(
        args.edges,
        layer_col=args.layer_col, start_col=args.start_col, end_col=args.end_col,
        id_base=args.id_base, has_header=args.has_header
    )
    union_pairs = sorted(list(build_union_pairs(data.edge_sets)))
    y = labels_for_layer(args.target_layer, union_pairs, data.edge_sets)
    idx = np.arange(len(union_pairs))
    tr, va, te = stratified_split(idx, y, *args.train_val_test, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}, f, indent=2)
    print(f"[freeze-splits] wrote {args.out}")


# ------------------------------ main ------------------------------

def main():
    parser = argparse.ArgumentParser(prog="lpmp", description="Multiplex link prediction")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Freeze-splits once and reuse everywhere
    # sp_freeze = sub.add_parser("freeze-splits", help="Create and save stratified pair splits")
    # add_common(sp_freeze)
    # sp_freeze.add_argument("--out", required=True, help="Path to write JSON splits {train,val,test}")
    # sp_freeze.set_defaults(func=run_freeze)

    # SLE
    sp_sle = sub.add_parser("sle", help="Transformer over layer-wise pair features (no GNN)")
    add_common_shared(sp_sle)
    add_embedding_args(sp_sle)
    sp_sle.add_argument("--sle-nhead", type=int, default=4)
    sp_sle.add_argument("--sle-layers", type=int, default=2)
    sp_sle.add_argument("--sle-ff", type=int, default=512)
    sp_sle.add_argument("--sle-dropout", type=float, default=0.1)
    sp_sle.add_argument("--include-target-in-context", action="store_true", default=True)
    sp_sle.set_defaults(func=run_sle)

    # GAT
    sp_gat = sub.add_parser("gat", help="Per-layer GAT + Transformer")
    add_common_shared(sp_gat)
    add_embedding_args(sp_gat)
    sp_gat.add_argument("--include-target-in-context", action="store_true", default=False)
    sp_gat.add_argument("--gat-heads", type=int, default=4)
    sp_gat.add_argument("--gat-dropout", type=float, default=0.1)
    sp_gat.add_argument("--tfm-layers", type=int, default=2)
    sp_gat.add_argument("--tfm-heads", type=int, default=4)
    sp_gat.add_argument("--tfm-ff", type=int, default=512)
    sp_gat.add_argument("--tfm-dropout", type=float, default=0.1)
    sp_gat.set_defaults(func=run_gat)



    # Baselines
    sp_mell = sub.add_parser("mell", help="MeLL baseline")
    add_common_shared(sp_mell)
    sp_mell.add_argument("--dim", type=int, default=128)
    sp_mell.set_defaults(func=run_mell)

    sp_hoplp = sub.add_parser("hoplp", help="HOPLP-MUL baseline")
    add_common_shared(sp_hoplp)
    sp_hoplp.add_argument("--K", type=int, default=3)
    sp_hoplp.add_argument("--alpha", type=float, default=0.5)
    sp_hoplp.add_argument("--weight-mode", choices=["density","uniform"], default="density")
    sp_hoplp.set_defaults(func=run_hoplp)

    sp_rmne = sub.add_parser("rmne", help="RMNE baseline")
    add_common_shared(sp_rmne)
    sp_rmne.add_argument("--dim", type=int, default=128)
    sp_rmne.add_argument("--K", type=int, default=10, help="Walks per node")
    sp_rmne.add_argument("--Le", type=int, default=40, help="Walk length")
    sp_rmne.add_argument("--window", type=int, default=5, help="Skip-gram window")
    sp_rmne.add_argument("--r", type=float, default=5.0)
    sp_rmne.add_argument("--t", type=float, default=5.0)
    sp_rmne.set_defaults(func=run_rmne)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    main()