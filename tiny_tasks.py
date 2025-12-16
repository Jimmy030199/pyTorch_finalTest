# -*- coding: utf-8 -*-
"""
tiny_tasks.py
- 合成資料 + TinyEncoder 驗證「有無位置編碼」的差異
- CLI:
    python tiny_tasks.py --task ends_equal --len 12 --n 6000 --epochs 10 --pe sincos
    python tiny_tasks.py --task compare_ij --i 1 --j 10 --len 12 --n 6000 --epochs 10 --pe sincos
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


def make_dataset_ends_equal(n: int, L: int, vocab: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    極對平衡版：一半樣本 y=1(首尾相等)，一半 y=0(首尾不等)
    """
    X = torch.randint(0, vocab, (n, L))
    y = torch.zeros(n, dtype=torch.long)

    half = n // 2

    # 正類：強制尾巴 = 頭
    X[:half, -1] = X[:half, 0]
    y[:half] = 1

    # 負類：強制尾巴 ≠ 頭
    for i in range(half, n):
        a = X[i, 0].item()
        b = random.randint(0, vocab - 2)
        if b >= a:
            b += 1
        X[i, -1] = b
        y[i] = 0

    # 打亂
    idx = torch.randperm(n)
    return X[idx], y[idx]


def make_dataset_compare_ij(
    n: int, L: int, vocab: int, i: int, j: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    y = 1 if X[:, i] > X[:, j] else 0
    """
    assert 0 <= i < L and 0 <= j < L and i != j
    X = torch.randint(0, vocab, (n, L))
    y = (X[:, i] > X[:, j]).long()
    return X, y


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



def main():
    parser = argparse.ArgumentParser(description="Tiny tasks for MHA + Positional Encoding")
    parser.add_argument("--task", type=str, choices=["ends_equal", "compare_ij"], default="ends_equal")
    parser.add_argument("--i", type=int, default=0, help="for compare_ij")
    parser.add_argument("--j", type=int, default=-1, help="for compare_ij (default L-1)")
    parser.add_argument("--len", type=int, default=12)
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--vocab", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pe", type=str, choices=["none", "sincos"], default="sincos")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L = args.len
    vocab = args.vocab
    n = args.n
    i = args.i
    j = args.j if args.j >= 0 else L - 1

    if args.task == "ends_equal":
        X, y = make_dataset_ends_equal(n, L, vocab)
    else:
        X, y = make_dataset_compare_ij(n, L, vocab, i=i, j=j)

    # train/val split
    n_train = int(0.8 * n)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:], y[n_train:]

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch_size, shuffle=False)

    cfg = TinyConfig(vocab=vocab, d_model=args.d_model, num_heads=args.heads, dropout=args.dropout)
    model = TinyEncoder(cfg).to(device)

    pe = None
    if args.pe == "sincos":
        pe = get_positional_encoding(L, cfg.d_model, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"[INFO] task={args.task} pe={args.pe} L={L} n={n} d_model={cfg.d_model} heads={cfg.num_heads}")

    for ep in range(1, args.epochs + 1):
        tl, ta = train_one(model, train_loader, optim, device, pe)
        vl, va = eval_one(model, val_loader, device, pe)
        print(f"[EP {ep:02d}] train loss={tl:.4f} acc={ta:.3f} | val loss={vl:.4f} acc={va:.3f}")


if __name__ == "__main__":
    main()
