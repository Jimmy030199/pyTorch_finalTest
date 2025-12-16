# -*- coding: utf-8 -*-

"""
mha.py
MultiHeadAttention 翻刻自 PyTorch 實作
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    # 套件式匯入
    from .attention import (
        scaled_dot_product_attention,
        make_causal_mask,
        combine_bool_masks,
    )
except Exception:
    # 直接執行檔案時的匯入
    from attention import (
        scaled_dot_product_attention,
        make_causal_mask,
        combine_bool_masks,
    )

class MultiHeadAttention(nn.Module):
    """
    多頭注意力（Encoder/Decoder 皆可用）
    介面：輸入 q,k,v 形狀 (B, L, D)；
          輸出 (B, L, D)，並可回傳 (B,H,L,L) 的注意力權重。
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = attn_dropout
        self.resid_drop = nn.Dropout(resid_dropout)

    # --------------------------
    # (B, L, D) → (B, H, L, Dk)
    # --------------------------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, L, D) → (B, H, L, Dk)
        """
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        return x

    # --------------------------
    # (B,H,L,Dk) → (B,L,D)
    # --------------------------
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, H, L, Dk) → (B, L, D)
        """
        B, H, L, Dk = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * Dk)
        return x

    # --------------------------
    # Forward
    # --------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # (B,1,L,L) 或 (1,1,L,L)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L)
        causal: bool = False,
        need_weights: bool = True,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, L, D = q.shape
        assert D == self.d_model and k.shape[-1] == D and v.shape[-1] == D

        # 線性投影
        q_proj = self.W_q(q)
        k_proj = self.W_k(k)
        v_proj = self.W_v(v)

        # 拆成多頭
        qh = self._split_heads(q_proj)   # (B,H,L,Dk)
        kh = self._split_heads(k_proj)
        vh = self._split_heads(v_proj)

        # ---------- 構建遮罩 ----------
        mask_bool = None

        # key padding mask
        if key_padding_mask is not None:
            # (B,1,1,L)
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(1)
            mask_bool = mask_k if mask_bool is None else (mask_bool | mask_k)

        # causal mask
        if causal:
            mask_c = make_causal_mask(L, device=q.device)  # (1,1,L,L)
            mask_bool = mask_c if mask_bool is None else (mask_bool | mask_c)

        # 合併 attn_mask 與 mask_bool
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            merged_mask = combine_bool_masks(mask_bool, attn_mask)
        else:
            merged_mask = mask_bool if attn_mask is None else attn_mask

        # ---------- 注意力 ----------
        out_h, attn = scaled_dot_product_attention(
            qh, kh, vh,
            mask=merged_mask,
            dropout_p=self.attn_dropout,
            training=self.training,
            return_weights=True,
        )

        # 合併多頭 → (B,L,D)
        out = self.W_o(self._merge_heads(out_h))
        out = self.resid_drop(out)

        if not need_weights:
            return out, None

        if average_attn_weights:
            attn_mean = attn.mean(dim=1)  # (B,L,L)
            return out, attn_mean

        return out, attn  # (B,H,L,L)


# -------------------------------------------------------------
# 測試用
# -------------------------------------------------------------
if __name__ == "__main__":
    B, L, D, H = 4, 10, 512, 8
    x = torch.randn(B, L, D)
    mha = MultiHeadAttention(D, H)
    y, w = mha(x, x, x, causal=False)
    print("y:", y.shape, "w:", None if w is None else w.shape)
