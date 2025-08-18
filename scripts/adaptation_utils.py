"""
Domain Adaptation Utilities for WiFi CSI models
- Adaptive BatchNorm (AdaBN) recalibration
- Gradient Reversal Layer (GRL) and DANN wrapper
- Few-shot fine-tuning helper

Usage examples (in your notebook or scripts):

from scripts.pytorch_deep_learning_model import WiFiSensingNet
from scripts.adaptation_utils import (
    recalibrate_bn, DomainAdversarialWrapper, train_dann,
    few_shot_adapt
)

# AdaBN on unlabeled target data
recalibrate_bn(model, unlabeled_loader, max_batches=200)

# DANN training
base = WiFiSensingNet(input_size=10800)
dann = DomainAdversarialWrapper(base, feature_dim=256, domain_hidden=128, num_domains=2).to(device)
train_dann(dann, source_loader, target_loader, class_criterion, domain_lambda=0.5, epochs=10, device=device)

# Few-shot fine-tuning on labeled target
few_shot_adapt(model, labeled_target_loader, steps=200, lr=1e-4, device=device)
"""

from __future__ import annotations
import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_)


class DomainAdversarialWrapper(nn.Module):
    """
    Wraps a classification model with a domain classifier head for DANN training.

    Assumptions about base_model:
    - Has attribute `network` which is nn.Sequential([... hidden layers ..., final_linear])
    - The last module is a Linear layer producing class logits.

    This wrapper splits base_model.network into:
    - feature_extractor = network[:-1]
    - label_head = network[-1]

    Then adds a domain_head MLP fed by features through a GRL.
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        domain_hidden: int = 128,
        num_domains: int = 2,
    ):
        super().__init__()
        assert hasattr(base_model, "network") and isinstance(base_model.network, nn.Sequential), (
            "base_model must have a .network nn.Sequential"
        )
        self.base = base_model
        # Split base network
        self.feature_extractor = nn.Sequential(*list(self.base.network.children())[:-1])
        self.label_head = list(self.base.network.children())[-1]
        if not isinstance(self.label_head, nn.Linear):
            raise ValueError("Expected final layer of base_model.network to be nn.Linear (class logits)")

        # Build domain head
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, domain_hidden),
            nn.BatchNorm1d(domain_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(domain_hidden, num_domains),
        )

        # Keep reference to class logits out dim
        self.num_classes = self.label_head.out_features

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shared features
        feats = self.feature_extractor(x)
        # Label logits
        class_logits = self.label_head(feats)
        # Domain logits via GRL
        rev = grad_reverse(feats, grl_lambda)
        domain_logits = self.domain_head(rev)
        return class_logits, domain_logits, feats


@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    unlabeled_loader: DataLoader,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """
    Adaptive BatchNorm (AdaBN): update running mean/var of BatchNorm layers using unlabeled target data.
    - No gradient steps, only BN running stats are updated by forward passes in train() mode.
    - Freezes all affine params during calibration to avoid weight drift.
    """
    was_training = model.training
    model.train()

    # Freeze affine params and all grads
    requires = {}
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            requires[m] = (m.weight.requires_grad if m.weight is not None else None,
                           m.bias.requires_grad if m.bias is not None else None)
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)

    batches = 0
    for xb, *rest in unlabeled_loader:
        xb = xb.to(device) if device is not None else xb
        _ = model(xb)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    # Restore requires_grad flags
    for m, (wreq, breq) in requires.items():
        if m.weight is not None and wreq is not None:
            m.weight.requires_grad_(wreq)
        if m.bias is not None and breq is not None:
            m.bias.requires_grad_(breq)

    if not was_training:
        model.eval()


def _poly_lambda(step: int, max_steps: int, init_lambda: float = 0.0, max_lambda: float = 1.0, power: float = 2.0) -> float:
    # Popular schedule from DANN: lambda gradually increases to 1
    p = step / max_steps
    return init_lambda + (max_lambda - init_lambda) * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0) ** power


def train_dann(
    model: DomainAdversarialWrapper,
    source_loader: DataLoader,
    target_loader: DataLoader,
    class_criterion: nn.Module,
    domain_lambda: float = 1.0,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    lambda_schedule: bool = True,
):
    """
    Train the DANN model with labeled source data and unlabeled target data.
    - class loss on source only
    - domain loss on both source and target (domain labels 0=source, 1=target)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    domain_criterion = nn.CrossEntropyLoss()

    # Ensure equal steps per epoch by cycling the shorter loader
    from itertools import cycle
    target_iter = cycle(target_loader)

    global_step = 0
    max_steps = epochs * len(source_loader)

    model.train()
    for epoch in range(epochs):
        running_cls, running_dom, total, correct = 0.0, 0.0, 0, 0
        for i, (xs, ys) in enumerate(source_loader):
            xt, = next(target_iter)[:1]
            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            # Lambda schedule for GRL strength
            if lambda_schedule:
                grl_lambda = _poly_lambda(global_step, max_steps, init_lambda=0.0, max_lambda=domain_lambda, power=1.0)
            else:
                grl_lambda = domain_lambda

            optimizer.zero_grad()

            # Forward: source for class+domain
            cls_src, dom_src, _ = model(xs, grl_lambda)
            # Forward: target for domain only (dummy class logits ignored)
            _, dom_tgt, _ = model(xt, grl_lambda)

            # Losses
            loss_cls = class_criterion(cls_src, ys)
            dom_labels_src = torch.zeros(dom_src.size(0), dtype=torch.long, device=device)
            dom_labels_tgt = torch.ones(dom_tgt.size(0), dtype=torch.long, device=device)
            loss_dom = domain_criterion(dom_src, dom_labels_src) + domain_criterion(dom_tgt, dom_labels_tgt)

            loss = loss_cls + loss_dom
            loss.backward()
            optimizer.step()

            running_cls += loss_cls.item()
            running_dom += loss_dom.item()
            with torch.no_grad():
                preds = cls_src.argmax(dim=1)
                correct += (preds == ys).sum().item()
                total += ys.size(0)

            global_step += 1

        print(f"Epoch {epoch+1}/{epochs} - cls_loss: {running_cls/len(source_loader):.4f} - dom_loss: {running_dom/len(source_loader):.4f} - src_acc: {100.0*correct/total:.2f}% - grl_lambda: {grl_lambda:.3f}")

    model.eval()


def few_shot_adapt(
    model: nn.Module,
    labeled_loader: DataLoader,
    steps: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    unfreeze_last_n_linear: int = 1,
):
    """
    Few-shot fine-tuning: freeze most of the network and fine-tune last N Linear layers on labeled target data.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last N Linear layers
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for m in linears[-unfreeze_last_n_linear:]:
        for p in m.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    it = 0
    while it < steps:
        for xb, yb in labeled_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            it += 1
            if it >= steps:
                break

    model.eval()
    return model
