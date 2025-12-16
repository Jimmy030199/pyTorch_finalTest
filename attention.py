# attention.py
# Scaled Dot-Product Attention
# 適用於 padding / causal 的遮罩

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn.functional as F

__all__ = [
    "scaled_dot_product_attention",
    "make_padding_mask",
    "make_causal_mask",
    "combine_bool_masks",
]


def _additive_mask_like(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    將布林遮罩轉為可與 scores 相加的加性遮罩（被遮位置 = -inf）
    """
    if mask.dtype != torch.bool:
        raise TypeError("mask 必須為 bool（True 表示要遮）")

    neg_inf = torch.finfo(scores.dtype).min

    return torch.zeros(1, dtype=scores.dtype, device=scores.device).masked_fill(
        torch.tensor(True, device=scores.device), neg_inf
    ) * mask.to(scores.dtype)   # 產生 0 或 -inf，並可與 scores 相加


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    return_weights: bool = True,
):
    """
    Scaled Dot-Product Attention
    參考形狀 q,k,v: (B, H, L, Dk) -> attn: (B, H, L, L), out: (B, H, L, Dk)

    Args:
        q, k, v: 形狀 (batch, heads, seq_len, d_k)
        mask: 可廣播到 (B, H, L, L) 的遮罩
              若為 bool，True 表示要遮
              若為浮點數則為加性遮罩（0 或 -inf）
        dropout_p: 注意力權重的 dropout 機率
        training: 是否為訓練模式
        return_weights: 是否回傳注意力權重

    Returns:
        output: (B, H, L, Dk)
        attn: (B, H, L, L)（若 return_weights=True）
    """

    B, H, L, Dk = q.shape
    scale = 1.0 / math.sqrt(Dk)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,L,L)

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask  # 期待為 0 / -inf 的加性遮罩

    attn = F.softmax(scores, dim=-1)  # 每一列和 = 1

    if dropout_p and training:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.matmul(attn, v)  # (B,H,L,Dk)

    if return_weights:
        return output, attn

    return output


def make_padding_mask(lengths: torch.Tensor, L: int) -> torch.Tensor:
    """
    建立 padding 遮罩（True=要遮）
    輸出形狀: (B, 1, 1, L)，供注意力中的 key 軸使用
    ----
    lengths: (B,)
    """
    device = lengths.device
    idxs = torch.arange(L, device=device).unsqueeze(0)   # (1, L)
    mask = idxs > lengths.unsqueeze(1)                   # (B, L)
    return mask.unsqueeze(1).unsqueeze(1)                # (B,1,1,L)


def make_causal_mask(L: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    建立 causal 遮罩（True=要遮未來）
    形狀: (1, 1, L, L) 的上三角（不含主對角）
    """
    m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    return m.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)


def combine_bool_masks(masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    將多個布林遮罩 OR 起來（None 會被忽略），回傳 None 或布林遮罩。
    注意：僅適用於 bool 遮罩；若需混合加性遮罩請自行處理。
    """
    valid = [m for m in masks if m is not None]
    if not valid:
        return None

    out = valid[0]
    for m in valid[1:]:
        out = out | m

    return out


if __name__ == "__main__":
    # 簡單宣測

    B, H, L, Dk = 2, 4, 5, 8
    q = torch.randn(B, H, L, Dk)
    k = torch.randn(B, H, L, Dk)
    v = torch.randn(B, H, L, Dk)

    out, attn = scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, training=False
    )

    print("out:", out.shape, "attn:", attn.shape)  # (2,4,5,8), (2,4,5,5)
