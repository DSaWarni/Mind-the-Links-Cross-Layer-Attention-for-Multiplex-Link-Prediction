# lpmp/train.py
from __future__ import annotations
from typing import Dict, Tuple, List, Any, Optional
import os, csv
import numpy as np
import torch
from torch.utils.data import DataLoader


# ----------------------------- small utils -----------------------------

def _to_device(obj: Any, device: torch.device):
    """Move tensors (recursively) to device; leave non-tensors as is."""
    if hasattr(obj, "to"):
        try:
            return obj.to(device, non_blocking=True)
        except Exception:
            return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(_to_device(v, device) for v in obj)
    return obj


def _approx_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Fast ROC AUC via rank statistic (WMW)."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(all_scores)) + 1
    # pos were first in concat
    sum_pos = ranks[: len(pos)].sum()
    auc = (sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    f1s = []
    for cls in (0, 1):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


def _scores_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute acc / macro-F1 / AUC from logits and binary targets."""
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y = targets.detach().cpu().numpy().reshape(-1).astype(int)
    yhat = (probs >= float(threshold)).astype(int)
    acc = float((yhat == y).mean())
    f1 = _macro_f1(y, yhat)
    auc = _approx_auc(y, probs)
    return {"acc": acc, "f1_macro": f1, "auc_approx": auc}


# -------------------------------- Trainer --------------------------------

class Trainer:
    """
    Generic trainer for pair scorers with BCEWithLogits loss.

    The model must implement: forward(target_layer: str, pair_idx: LongTensor[B,2]) -> logits[B,1]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[str] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        num_workers: int = 0,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.num_workers = int(num_workers)

    def _make_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        target_layer: str,
        train: bool,
        pos_weight: Optional[torch.Tensor],
    ) -> Tuple[float, Dict[str, float], torch.Tensor, torch.Tensor]:
        self.model.train(train)
        total_loss = 0.0
        n_batches = 0

        all_logits = []
        all_targets = []

        if pos_weight is not None:
            crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        else:
            crit = torch.nn.BCEWithLogitsLoss()

        for batch in loader:
            batch = _to_device(batch, self.device)
            with torch.set_grad_enabled(train):
                logits = self.model(target_layer, batch["pair_idx"])  # (B,1)
                loss = crit(logits, batch["y"])
                if train:
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.opt.step()

            total_loss += float(loss.detach().item())
            n_batches += 1
            all_logits.append(logits.detach())
            all_targets.append(batch["y"].detach())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        scores = _scores_from_logits(logits, targets, threshold=0.5)
        return total_loss / max(1, n_batches), scores, logits, targets
    
    def fit(
        self,
        target_layer: str,
        train_set,
        val_set,
        test_set,
        batch_size: int = 512,
        pos_weight_train: Optional[float | torch.Tensor] = None,
        pos_weight_eval: Optional[float | torch.Tensor] = None,
        epochs: int = 30,
        log_every: int = 1,
        # new:
        csv_path: Optional[str] = None,
        early_stopping_patience: int = 0,  # 0 = disabled
    ) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        """
        Train, choose threshold on validation (max macro-F1), early-stop on val macro-F1,
        then report test with that frozen threshold.
        """
        tr_loader = self._make_loader(train_set, batch_size, shuffle=True)
        va_loader = self._make_loader(val_set, batch_size, shuffle=False)
        te_loader = self._make_loader(test_set, batch_size, shuffle=False)

        if isinstance(pos_weight_train, (int, float)):
            pos_weight_train = torch.tensor(float(pos_weight_train), dtype=torch.float32).view(1)
        if isinstance(pos_weight_eval, (int, float)):
            pos_weight_eval = torch.tensor(float(pos_weight_eval), dtype=torch.float32).view(1)

        # --- CSV logger (correctly manage the file handle) ---
        csv_fh = None
        csv_writer = None
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            csv_fh = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_fh)
            csv_writer.writerow([
                "epoch",
                "train_loss", "train_f1_macro", "train_acc", "train_auc",
                "val_loss", "val_f1_macro@0.5", "val_acc@0.5", "val_auc",
                "best_val_f1", "best_thr",
            ])
            csv_fh.flush()

        history: List[Dict[str, float]] = []
        best_val_f1 = -1.0
        best_state = None
        best_thr = 0.5
        patience_left = int(early_stopping_patience)

        thr_grid = np.linspace(0.05, 0.95, 19)

        for ep in range(1, epochs + 1):
            tr_loss, tr_scores, _, _ = self._run_epoch(tr_loader, target_layer, True, pos_weight_train)
            va_loss, va_scores0, va_logits, va_targets = self._run_epoch(va_loader, target_layer, False, pos_weight_eval)

            # sweep thresholds on validation to maximize macro-F1
            cur_best_thr, cur_best_f1 = best_thr, va_scores0["f1_macro"]
            for thr in thr_grid:
                s = _scores_from_logits(va_logits, va_targets, threshold=float(thr))
                if s["f1_macro"] > cur_best_f1:
                    cur_best_f1 = s["f1_macro"]
                    cur_best_thr = float(thr)

            improved = cur_best_f1 > best_val_f1
            if improved:
                best_val_f1 = cur_best_f1
                best_thr = cur_best_thr
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                patience_left = int(early_stopping_patience)
            else:
                if early_stopping_patience > 0:
                    patience_left -= 1

            if (ep % max(1, log_every)) == 0:
                print(
                    f"[epoch {ep:03d}] "
                    f"train: loss={tr_loss:.4f} f1={tr_scores['f1_macro']:.3f} "
                    f"| val: loss={va_loss:.4f} f1@0.5={va_scores0['f1_macro']:.3f} "
                    f"best_val_f1={best_val_f1:.3f} thr={best_thr:.2f} "
                    f"(patience_left={patience_left})",
                    flush=True,
                )

            row = {
                "epoch": ep,
                "train_loss": tr_loss,
                "train_f1_macro": tr_scores["f1_macro"],
                "train_acc": tr_scores["acc"],
                "train_auc": tr_scores["auc_approx"],
                "val_loss": va_loss,
                "val_f1_macro@0.5": va_scores0["f1_macro"],
                "val_acc@0.5": va_scores0["acc"],
                "val_auc": va_scores0["auc_approx"],
                "best_val_f1": best_val_f1,
                "best_thr": best_thr,
            }
            history.append(row)

            if csv_writer is not None:
                csv_writer.writerow([
                    row["epoch"], row["train_loss"], row["train_f1_macro"], row["train_acc"], row["train_auc"],
                    row["val_loss"], row["val_f1_macro@0.5"], row["val_acc@0.5"], row["val_auc"],
                    row["best_val_f1"], row["best_thr"],
                ])
                csv_fh.flush()

            # Early stopping check
            if early_stopping_patience > 0 and patience_left <= 0:
                print(f"[early stopping] no val F1 improvement for {early_stopping_patience} epochs.", flush=True)
                break

        # load best weights (by val F1) before final test
        if best_state is not None:
            self.model.load_state_dict(best_state)

        te_loss, _, te_logits, te_targets = self._run_epoch(te_loader, target_layer, False, pos_weight_eval)
        te_scores = _scores_from_logits(te_logits, te_targets, threshold=best_thr)

        print(
            f"[TEST] loss={te_loss:.4f} "
            f"f1={te_scores['f1_macro']:.3f} acc={te_scores['acc']:.3f} auc={te_scores['auc_approx']:.3f} "
            f"thr={best_thr:.2f}",
            flush=True,
        )

        # Append a final TEST row and close CSV cleanly
        if csv_writer is not None:
            csv_writer.writerow([
                "TEST",
                te_loss, te_scores["f1_macro"], te_scores["acc"], te_scores["auc_approx"],
                "", "", "", "",
                best_val_f1, best_thr,
            ])
            csv_fh.close()

        test_out = {"test_loss": te_loss, "threshold": float(best_thr), **te_scores}
        return history, test_out
