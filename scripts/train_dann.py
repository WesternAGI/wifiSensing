#!/usr/bin/env python3
"""
Training CLI for WiFi CSI with baselines: standard, AdaBN, DANN, few-shot.

Example usages:

1) Standard supervised on source then evaluate on eval set
   python scripts/train_dann.py \
     --baseline standard \
     --source-x data/source_X.pt --source-y data/source_y.pt \
     --eval-x data/eval_X.pt --eval-y data/eval_y.pt

2) AdaBN: train on source, recalibrate BN on unlabeled target, evaluate on eval set
   python scripts/train_dann.py \
     --baseline adabn \
     --source-x data/source_X.pt --source-y data/source_y.pt \
     --target-x-unlabeled data/target_unlabeled_X.pt \
     --eval-x data/eval_X.pt --eval-y data/eval_y.pt

3) DANN: adversarial with labeled source + unlabeled target, evaluate on eval set
   python scripts/train_dann.py \
     --baseline dann \
     --source-x data/source_X.pt --source-y data/source_y.pt \
     --target-x-unlabeled data/target_unlabeled_X.pt \
     --eval-x data/eval_X.pt --eval-y data/eval_y.pt

4) Few-shot: train on source, then fine-tune last layer(s) on small labeled target, evaluate
   python scripts/train_dann.py \
     --baseline fewshot \
     --source-x data/source_X.pt --source-y data/source_y.pt \
     --target-x data/target_fewshot_X.pt --target-y data/target_fewshot_y.pt \
     --eval-x data/eval_X.pt --eval-y data/eval_y.pt

Inputs can be .pt (torch.save tensor/array) or .npy (NumPy array). Shapes:
- X: (N, 10800)
- y: (N,) integers {0,1}
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scripts.pytorch_deep_learning_model import WiFiSensingNet, WiFiSensingNetAttn
from scripts.adaptation_utils import (
    DomainAdversarialWrapper, train_dann, recalibrate_bn, few_shot_adapt
)


def load_array(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        arr = torch.load(path)
        return torch.as_tensor(arr).float() if arr.dtype != torch.long else torch.as_tensor(arr)
    elif ext == ".npy":
        arr = np.load(path)
        return torch.from_numpy(arr)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def make_loader(X: torch.Tensor, y: Optional[torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    if y is None:
        ds = TensorDataset(X)
    else:
        y = y.long()
        ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_standard(model: nn.Module, train_loader: DataLoader, epochs: int, device: torch.device, lr: float, weight_decay: float = 1e-5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        print(f"Epoch {ep+1}/{epochs} - loss {loss_sum/len(train_loader):.4f} - acc {100.0*correct/total:.2f}%")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, name: str = "Eval"):
    model.eval()
    total, correct = 0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
            else:
                xb, = batch
                xb = xb.to(device)
                yb = None
            logits = model(xb)
            pred = logits.argmax(1)
            if yb is not None:
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                all_true.extend(yb.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    if total > 0:
        print(f"{name} accuracy: {100.0*correct/total:.2f}%")
    return np.array(all_true) if total > 0 else None, np.array(all_pred)


def main():
    ap = argparse.ArgumentParser(description="WiFi CSI Baselines: standard, AdaBN, DANN, few-shot")
    ap.add_argument("--baseline", required=True, choices=["standard", "adabn", "dann", "fewshot"], help="Select training baseline")
    ap.add_argument("--attn", action="store_true", help="Use attention-augmented model")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--feature-dim", type=int, default=256, help="Feature dim feeding domain head (for DANN)")
    ap.add_argument("--domain-hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Data paths
    ap.add_argument("--source-x", required=True, help="Source X (.pt or .npy)")
    ap.add_argument("--source-y", required=True, help="Source y (.pt or .npy)")
    ap.add_argument("--target-x-unlabeled", help="Unlabeled target X for AdaBN/DANN")
    ap.add_argument("--target-x", help="Labeled target X for few-shot")
    ap.add_argument("--target-y", help="Labeled target y for few-shot")
    ap.add_argument("--eval-x", help="Evaluation X (.pt or .npy)")
    ap.add_argument("--eval-y", help="Evaluation y (.pt or .npy)")

    args = ap.parse_args()
    device = torch.device(args.device)

    # Load data
    Xs = load_array(args.source_x).float()
    ys = load_array(args.source_y).long()

    X_eval = load_array(args.eval_x).float() if args.eval_x else None
    y_eval = load_array(args.eval_y).long() if args.eval_y else None

    tgt_unlab = load_array(args.target_x_unlabeled).float() if args.target_x_unlabeled else None
    Xt = load_array(args.target_x).float() if args.target_x else None
    yt = load_array(args.target_y).long() if args.target_y else None

    train_loader = make_loader(Xs, ys, args.batch_size, shuffle=True)
    eval_loader = make_loader(X_eval, y_eval, args.batch_size, shuffle=False) if X_eval is not None else None

    # Model
    if args.attn:
        model = WiFiSensingNetAttn(input_size=Xs.shape[1])
    else:
        model = WiFiSensingNet(input_size=Xs.shape[1], hidden_sizes=[2048, 1024, 512, 256], num_classes=2)

    if args.baseline == "standard":
        train_standard(model, train_loader, args.epochs, device, args.lr)
        if eval_loader:
            evaluate(model.to(device), eval_loader, device, name="Eval")

    elif args.baseline == "adabn":
        if tgt_unlab is None:
            raise ValueError("--target-x-unlabeled is required for AdaBN baseline")
        train_standard(model, train_loader, args.epochs, device, args.lr)
        unlab_loader = make_loader(tgt_unlab, None, args.batch_size, shuffle=False)
        recalibrate_bn(model.to(device), unlab_loader, max_batches=200, device=device)
        if eval_loader:
            evaluate(model, eval_loader, device, name="Eval (after AdaBN)")

    elif args.baseline == "dann":
        if tgt_unlab is None:
            raise ValueError("--target-x-unlabeled is required for DANN baseline")
        # Wrap with domain head
        dann = DomainAdversarialWrapper(model, feature_dim=args.feature_dim, domain_hidden=args.domain_hidden, num_domains=2).to(device)
        source_loader = train_loader
        target_loader = make_loader(tgt_unlab, None, args.batch_size, shuffle=True)
        class_criterion = nn.CrossEntropyLoss()
        train_dann(dann, source_loader, target_loader, class_criterion, domain_lambda=1.0, epochs=args.epochs, lr=args.lr, device=device)
        if eval_loader:
            # Use the base classification branch for eval
            evaluate(dann.base.to(device), eval_loader, device, name="Eval (DANN)")

    elif args.baseline == "fewshot":
        if Xt is None or yt is None:
            raise ValueError("--target-x and --target-y are required for few-shot baseline")
        train_standard(model, train_loader, max(1, args.epochs // 2), device, args.lr)
        labeled_loader = make_loader(Xt, yt, args.batch_size, shuffle=True)
        few_shot_adapt(model.to(device), labeled_loader, steps=200, lr=1e-4, device=device, unfreeze_last_n_linear=1)
        if eval_loader:
            evaluate(model, eval_loader, device, name="Eval (few-shot)")

    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")


if __name__ == "__main__":
    main()
