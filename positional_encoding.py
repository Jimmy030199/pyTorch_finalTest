# -*- coding: utf-8 -*-
"""
positional_encoding.py
Sinusoidal 位置編碼（函式與 nn.Module 版本）
"""

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn

__all__ = ["get_positional_encoding", "SinusoidalPositionalEncoding"]


def get_positional_encoding(
    max_len: int,
    d_model: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    回傳形狀 (1, max_len, d_model) 的 Sin/Cos 位置編碼，供 x + pe 使用
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    pe = torch.zeros(max_len, d_model, dtype=dtype, device=device)
    position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=dtype, device=device)
        * (-(math.log(10000.0) / d_model))
    )  # (D/2,)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, L, D)


class SinusoidalPositionalEncoding(nn.Module):
    """
    與 get_positional_encoding 等價的模組版本（會緩存到 buffer）
    """

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = get_positional_encoding(max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        x = x + self.pe[:, :L, :].to(dtype=x.dtype)
        return self.dropout(x)


if __name__ == "__main__":
    pe = get_positional_encoding(8, 6)
    print(pe.shape)  # (1, 8, 6)