#!/usr/bin/env python3
"""
Train compact and standard CNNs on a merged dataset with any number of
classes, using an event-level split.

Usage (from repo root):
  python scripts/train_multiclass.py --dataset notebooks/02_labeling/labeled_data/dataset_4class_<ts>
  python scripts/train_multiclass.py --dataset <dir> --models standard --epochs 80
  python scripts/train_multiclass.py --dataset <dir> --classes Noise,Earthquake,Blast   # drop a class

Differences from notebooks/03_training/train_cnn_multiclass.ipynb:
  * split by event id BEFORE cropping, so no crop of a window (or another
    station's recording of the same event) leaks across train/val/test;
  * class list comes from the dataset's classes.json, not a hardcoded map;
  * per-network metrics on the test set (AK vs AM) so the Raspberry Shake
    number is reported separately from the broadband number;
  * checkpoint stores class_names, architecture and preprocessing so
    scripts/package_model.py can write CLUE metadata without guessing.

Outputs in --out (default models/):
  seismic_cnn_<arch>_<dataset>_<ts>.pth, training_summary_<arch>_<ts>.txt,
  metrics_<arch>_<ts>.json, confusion_<arch>_<ts>.png (if matplotlib).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models import get_model  # noqa: E402
from src.data.preprocessing import DataAugmentation  # noqa: E402
from src.data.windows import (load_dataset, event_level_split, expand_crops,  # noqa: E402
                              class_weights_inverse_freq)

DEFAULT_LR = {"compact": 5e-4, "standard": 4e-4}


class CropDataset(Dataset):
    def __init__(self, X, y, augment=None):
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (n, 1, L)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment is not None:
            x = torch.from_numpy(self.augment(x.numpy()).astype(np.float32))
        return x, self.y[i]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, max_grad_norm=2.0):
    train = optimizer is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            tot_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return tot_loss / max(n, 1), 100.0 * correct / max(n, 1)


def predict(model, X, device, batch=256):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(device)
            p = torch.softmax(model(xb), dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(1))
    if not preds:
        return np.zeros(0, int), np.zeros((0, 0))
    return np.concatenate(preds), np.concatenate(probs)


def per_class_recall(y_true, y_pred, num_classes):
    out = []
    for c in range(num_classes):
        m = y_true == c
        out.append(float((y_pred[m] == c).mean()) if m.any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["compact", "standard"], choices=["compact", "standard"])
    ap.add_argument("--classes", default=None,
                    help="comma-separated subset/order of classes to train on (others dropped)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="override per-model default")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--onset-crops", type=int, default=1)
    ap.add_argument("--free-crops", type=int, default=1)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--split", default="0.7,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "models")
    ap.add_argument("--tag", default=None, help="dataset tag used in output filenames")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    # ---- data -------------------------------------------------------------
    X, y, meta, info = load_dataset(args.dataset)
    classes = list(info["classes"])
    if args.classes:
        keep_names = [c.strip() for c in args.classes.split(",")]
        remap = {classes.index(c): i for i, c in enumerate(keep_names)}
        mask = np.isin(y, list(remap))
        X, y, meta = X[mask], np.array([remap[int(v)] for v in y[mask]]), meta[mask].reset_index(drop=True)
        classes = keep_names
    num_classes = len(classes)
    if "Noise" not in classes:
        print("WARNING: no class named 'Noise'; CLUE relies on that exact name for event filtering.")

    window = info.get("window") or {}
    fs = float(window.get("sampling_rate", 100))
    target_len = int(round(float(window.get("train_len_sec", 60)) * fs))
    onset = meta["onset_sample"].values if "onset_sample" in meta.columns else np.full(len(y), np.nan)

    ratios = tuple(float(r) for r in args.split.split(","))
    group = meta["event_id"].astype(str).values if "event_id" in meta.columns else np.arange(len(y))
    if "event_id" not in meta.columns:
        print("WARNING: metadata has no event_id; falling back to a per-window split (leaky).")
    masks = event_level_split(group, ratios, seed=args.seed, strata=y)

    splits = {}
    for k, (name, m) in enumerate(masks.items()):
        idx = np.where(m)[0]
        Xs, ys, src = expand_crops([X[i] for i in idx], y[idx], onset[idx], target_len,
                                   args.onset_crops if name == "train" else 1,
                                   args.free_crops if name == "train" else 0,
                                   seed=args.seed + k)
        splits[name] = (Xs, ys, idx[src] if len(src) else src)
        counts = {classes[c]: int((ys == c).sum()) for c in range(num_classes)}
        print(f"  {name:5s}: {len(idx):5d} windows from {len(np.unique(group[idx])):4d} events -> "
              f"{len(ys):5d} crops {counts}")

    Xtr, ytr, _ = splits["train"]
    Xva, yva, _ = splits["val"]
    Xte, yte, te_src = splits["test"]
    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        sys.exit("One of the splits is empty; need more events.")

    augment = None if args.no_augment else DataAugmentation(noise_level=0.05, time_shift_range=200,
                                                              amplitude_scale_range=(0.7, 1.3))
    train_loader = DataLoader(CropDataset(Xtr, ytr, augment), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CropDataset(Xva, yva), batch_size=args.batch_size)

    weights = None if args.no_class_weights else torch.tensor(
        class_weights_inverse_freq(ytr, num_classes), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or args.dataset.name
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}

    # ---- train ------------------------------------------------------------
    for arch in args.models:
        print(f"\n=== {arch} ===")
        model = get_model(arch, num_classes=num_classes, input_channels=1, input_length=target_len).to(device)
        lr = args.lr or DEFAULT_LR[arch]
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-7)
        print(f"  parameters: {model.count_parameters():,}  lr={lr}")

        best, best_state, bad, history = float("inf"), None, 0, []
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step(va_loss)
            history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
            print(f"  ep {epoch:3d}  train {tr_loss:.4f}/{tr_acc:5.1f}%  val {va_loss:.4f}/{va_acc:5.1f}%")
            if va_loss < best - 1e-3:
                best, bad = va_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  early stop at epoch {epoch}")
                    break
        model.load_state_dict(best_state)

        # ---- evaluate -----------------------------------------------------
        y_pred, _ = predict(model, Xte, device)
        report = classification_report(yte, y_pred, labels=list(range(num_classes)),
                                       target_names=classes, digits=3, zero_division=0)
        cm = confusion_matrix(yte, y_pred, labels=list(range(num_classes)))
        recall = per_class_recall(yte, y_pred, num_classes)
        acc = float((y_pred == yte).mean())
        by_network = {}
        if "network" in meta.columns and len(te_src):
            nets = meta["network"].values[te_src]
            for net in np.unique(nets):
                m = nets == net
                by_network[str(net)] = dict(n=int(m.sum()), accuracy=float((y_pred[m] == yte[m]).mean()),
                                            recall=per_class_recall(yte[m], y_pred[m], num_classes))

        metrics = dict(arch=arch, classes=classes, test_accuracy=acc, test_recall=recall,
                       confusion=cm.tolist(), by_network=by_network, best_val_loss=best,
                       epochs_run=len(history), n_train=int(len(ytr)), n_val=int(len(yva)),
                       n_test=int(len(yte)), history=history, dataset=str(args.dataset),
                       train_seconds=time.time() - t0)
        results[arch] = metrics

        ckpt_path = args.out / f"seismic_cnn_{arch}_{tag}_{ts}.pth"
        torch.save(dict(model_state_dict=best_state, architecture=arch, class_names=classes,
                        num_classes=num_classes, input_channels=1, input_length=target_len,
                        sampling_rate=fs, window=window, dataset=str(args.dataset),
                        metrics={k: v for k, v in metrics.items() if k != "history"},
                        parameters=model.count_parameters(), created=ts), ckpt_path)
        with open(args.out / f"metrics_{arch}_{ts}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(args.out / f"training_summary_{arch}_{ts}.txt", "w") as f:
            f.write(f"Seismic CNN Training Summary - {arch.upper()} Model\n{'=' * 60}\n\n")
            f.write(f"Date: {ts}\nDataset: {args.dataset}\nClasses: {classes}\n")
            f.write(f"Parameters: {model.count_parameters():,}\nInput shape: (batch, 1, {target_len})\n\n")
            f.write(f"Split (event-level): train {len(ytr)} / val {len(yva)} / test {len(yte)} crops\n")
            f.write(f"Epochs run: {len(history)}  best val loss: {best:.4f}\n\n")
            f.write(f"Test accuracy: {acc * 100:.2f}%\n\nClassification report (test):\n{report}\n")
            f.write("Confusion matrix (rows=true, cols=pred):\n")
            f.write("  " + "  ".join(f"{c[:8]:>8s}" for c in classes) + "\n")
            for c, row in zip(classes, cm):
                f.write(f"  {c[:8]:>8s} " + "  ".join(f"{v:8d}" for v in row) + "\n")
            if by_network:
                f.write("\nPer-network test accuracy:\n")
                for net, d in by_network.items():
                    f.write(f"  {net}: n={d['n']} acc={d['accuracy'] * 100:.1f}% recall="
                            f"{[round(r, 3) if r == r else None for r in d['recall']]}\n")
        print(f"\n{report}")
        print(f"  saved {ckpt_path.name}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(1.2 * num_classes + 3, 1.2 * num_classes + 2))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(num_classes)); ax.set_xticklabels(classes, rotation=45, ha="right")
            ax.set_yticks(range(num_classes)); ax.set_yticklabels(classes)
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{arch}: test acc {acc * 100:.1f}%")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(args.out / f"confusion_{arch}_{ts}.png", dpi=120)
            plt.close(fig)
        except Exception:  # noqa: BLE001 - plotting is optional
            pass

    # ---- acceptance check ---------------------------------------------------
    print("\nAcceptance (per-class recall >= 0.80 on test):")
    for arch, m in results.items():
        flags = ["OK" if r == r and r >= 0.8 else "LOW" for r in m["test_recall"]]
        print(f"  {arch:9s} " + "  ".join(f"{c}={r:.2f}({f})" for c, r, f in zip(classes, m["test_recall"], flags)))
        for net, d in m["by_network"].items():
            print(f"    {net}: acc={d['accuracy']:.2f} n={d['n']}")


if __name__ == "__main__":
    main()
