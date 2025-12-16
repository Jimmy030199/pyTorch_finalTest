# -*- coding: utf-8 -*-
"""
tiny_tasks.py
- 合成資料 + TinyEncoder 驗證「有無位置編碼」的差異
"""

from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from .mha import MultiHeadAttention
    from .positional_encoding import get_positional_encoding
except Exception:
    from mha import MultiHeadAttention
    from positional_encoding import get_positional_encoding


def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
#                   Dataset Builders
# ============================================================

def make_dataset_ends_equal(n: int, L: int, vocab: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randint(0, vocab, (n, L))
    y = torch.zeros(n, dtype=torch.long)

    half = n // 2

    X[:half, -1] = X[:half, 0]
    y[:half] = 1

    for i in range(half, n):
        a = X[i, 0].item()
        b = random.randint(0, vocab - 2)
        if b >= a:
            b += 1
        X[i, -1] = b
        y[i] = 0

    idx = torch.randperm(n)
    return X[idx], y[idx]


def make_dataset_compare_ij(
    n: int, L: int, vocab: int, i: int, j: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 0 <= i < L and 0 <= j < L and i != j
    X = torch.randint(0, vocab, (n, L))
    y = (X[:, i] > X[:, j]).long()
    return X, y


# ============================================================
#                   Tiny Encoder Model
# ============================================================

@dataclass
class TinyConfig:
    vocab: int = 100
    d_model: int = 128
    num_heads: int = 4
    dropout: float = 0.0


class TinyEncoder(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.mha = MultiHeadAttention(cfg.d_model, cfg.num_heads,
                                      attn_dropout=cfg.dropout,
                                      resid_dropout=cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.cls = nn.Linear(cfg.d_model, 2)

    def forward(self, x_ids: torch.Tensor, pe: Optional[torch.Tensor] = None):
        h = self.emb(x_ids)  # (B,L,D)

        if pe is not None:
            h = h + pe[:, :h.size(1), :].to(h.dtype)

        y, attn = self.mha(h, h, h, need_weights=True)
        h = self.norm(h + y)

        pooled = h.mean(dim=1)  # (B,D)
        return self.cls(pooled), attn


# ============================================================
#                   Train / Eval
# ============================================================

def train_one(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pe: Optional[torch.Tensor]
) -> Tuple[float, float]:

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits, _ = model(xb, pe=pe)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_one(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pe: Optional[torch.Tensor]
) -> Tuple[float, float]:

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits, _ = model(xb, pe=pe)
        loss = loss_fn(logits, yb)

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, total_correct / total


# ============================================================
#        🎯 作業重點：學生版（要比較 use_pe=True/False）
# ============================================================

def run_tiny_task_student_version(
    task: str = "ends_equal",
    L: int = 12,
    n: int = 6000,
    vocab: int = 100,
    d_model: int = 128,
    heads: int = 4,
    dropout: float = 0.0,
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 3e-3,
    i: int = 0,
    j: int = -1,
    use_pe: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. build dataset
    if task == "ends_equal":
        X, y = make_dataset_ends_equal(n, L, vocab)
    elif task == "compare_ij":
        if j < 0:
            j = L - 1
        X, y = make_dataset_compare_ij(n, L, vocab, i=i, j=j)
    else:
        raise ValueError("Unknown task")

    n_train = int(n * 0.8)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:], y[n_train:]

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)

    # 2. build model
    cfg = TinyConfig(vocab=vocab, d_model=d_model, num_heads=heads, dropout=dropout)
    model = TinyEncoder(cfg).to(device)

    # 3. PE
    pe = None
    if use_pe:
        pe = get_positional_encoding(L, d_model, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[INFO] task={task} use_pe={use_pe}")

    # 4. training loop
    for ep in range(1, epochs + 1):
        tl, ta = train_one(model, train_loader, optim, device, pe)
        vl, va = eval_one(model, val_loader, device, pe)

        print(f"[EP {ep:02d}] train loss={tl:.4f} acc={ta:.3f} | "
              f"val loss={vl:.4f} acc={va:.3f}")

    return model


# ============================================================
#                    main()（保留）
# ============================================================

def main():
    print("This main() is kept for compatibility but not used for HW.")


# ============================================================
#           🔥🔥 老師要求的最終執行（作業要交的）
# ============================================================

if __name__ == "__main__":

    print("\n====== 無位置編碼 use_pe=False ======")
    run_tiny_task_student_version(task="ends_equal", use_pe=False)

    print("\n====== 有位置編碼 use_pe=True ======")
    run_tiny_task_student_version(task="ends_equal", use_pe=True)
