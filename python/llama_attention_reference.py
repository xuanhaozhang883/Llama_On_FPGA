#!/usr/bin/env python3
"""第0层标准FP32与RTL位感知GQA Attention参考实现。

公开张量布局统一为 ``[head, token, head_dim]``。标准路径用于验证
Attention算法语义；RTL位感知路径复用当前RTL对应的逐阶段舍入模型，
两者是彼此独立的验收参考。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from python.flash_attention_tile_model import (
    flash_rtl_exact,
    qk_score,
    rope_vector,
)


def float_to_bf16_words(values: object) -> np.ndarray:
    """将数值转换为BF16 RNE位模式，返回形状不变的``uint16``数组。"""

    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + rounding_bias) >> 16).astype(np.uint16)


def bf16_words_to_float(words: np.ndarray) -> np.ndarray:
    """把BF16 ``uint16``位模式精确展开为FP32。"""

    array = np.asarray(words)
    if array.dtype != np.uint16:
        raise ValueError("BF16 words必须使用uint16数组")
    bits = array.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _validate_rope_tensors(
    values: torch.Tensor,
    sine: torch.Tensor,
    cosine: torch.Tensor,
) -> tuple[int, int]:
    if not isinstance(values, torch.Tensor):
        raise TypeError("RoPE输入必须是torch.Tensor")
    if not isinstance(sine, torch.Tensor) or not isinstance(cosine, torch.Tensor):
        raise TypeError("RoPE sin/cos必须是torch.Tensor")
    if values.dtype != torch.float32 or sine.dtype != torch.float32 or cosine.dtype != torch.float32:
        raise ValueError("标准Attention的RoPE输入必须为FP32")
    if values.ndim < 2:
        raise ValueError("RoPE输入形状必须至少包含token和head_dim")

    seq_len, head_dim = values.shape[-2:]
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim必须是正偶数")
    expected = (seq_len, head_dim // 2)
    if tuple(sine.shape) != expected or tuple(cosine.shape) != expected:
        raise ValueError(f"RoPE sin/cos形状必须为{expected}")
    return seq_len, head_dim


def apply_split_half_rope(
    values: torch.Tensor,
    sine: torch.Tensor,
    cosine: torch.Tensor,
) -> torch.Tensor:
    """应用Llama split-half RoPE：``[x1,x2] -> [-x2,x1]``。"""

    seq_len, head_dim = _validate_rope_tensors(values, sine, cosine)
    half = head_dim // 2
    broadcast_shape = (1,) * (values.ndim - 2) + (seq_len, half)
    sin_values = sine.reshape(broadcast_shape)
    cos_values = cosine.reshape(broadcast_shape)
    first = values[..., :half]
    second = values[..., half:]
    return torch.cat(
        (first * cos_values - second * sin_values,
         first * sin_values + second * cos_values),
        dim=-1,
    )


def _validate_standard_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sine: torch.Tensor,
    cosine: torch.Tensor,
) -> tuple[int, int, int, int]:
    tensors = (q, k, v, sine, cosine)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("标准Attention输入必须全部是torch.Tensor")
    if any(value.dtype != torch.float32 for value in tensors):
        raise ValueError("标准Attention输入必须全部为FP32")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q/K/V形状必须为[head,token,head_dim]")
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("K/V形状必须完全相同")

    q_heads, seq_len, head_dim = q.shape
    kv_heads, kv_seq_len, kv_head_dim = k.shape
    if head_dim <= 0 or head_dim % 2 or kv_head_dim != head_dim:
        raise ValueError("Q/K/V的head_dim必须相同且为正偶数")
    if seq_len <= 0 or kv_seq_len != seq_len:
        raise ValueError("Q/K/V的token形状必须相同且非空")
    if kv_heads <= 0 or q_heads <= 0 or q_heads % kv_heads:
        raise ValueError("Q头数必须能被KV头数整除")
    if tuple(sine.shape) != (seq_len, head_dim // 2) or tuple(cosine.shape) != tuple(sine.shape):
        raise ValueError("RoPE sin/cos形状与Q/K不匹配")
    if not all(torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("Attention输入不能包含NaN或无穷值")
    return q_heads, kv_heads, seq_len, head_dim


def standard_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sine: torch.Tensor,
    cosine: torch.Tensor,
) -> torch.Tensor:
    """计算带split-half RoPE和因果Mask的标准CPU FP32 GQA Attention。"""

    q_heads, kv_heads, seq_len, head_dim = _validate_standard_inputs(
        q, k, v, sine, cosine)
    q_rotated = apply_split_half_rope(q, sine, cosine)
    k_rotated = apply_split_half_rope(k, sine, cosine)
    group_size = q_heads // kv_heads
    kv_indices = torch.arange(q_heads, device=q.device) // group_size
    expanded_k = k_rotated.index_select(0, kv_indices)
    expanded_v = v.index_select(0, kv_indices)

    scores = torch.matmul(q_rotated, expanded_k.transpose(-1, -2))
    scores = scores * (head_dim ** -0.5)
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, expanded_v)


def _validate_rtl_inputs(
    q_words: np.ndarray,
    k_words: np.ndarray,
    v_words: np.ndarray,
    sine_words: np.ndarray,
    cosine_words: np.ndarray,
    lut: Sequence[int],
    tile: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value) for value in (
        q_words, k_words, v_words, sine_words, cosine_words))
    if any(value.dtype != np.uint16 for value in arrays):
        raise ValueError("RTL位感知输入必须全部是uint16 BF16位模式")
    q, k, v, sine, cosine = arrays
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q/K/V形状必须为[head,token,head_dim]")
    if k.shape != v.shape:
        raise ValueError("K/V形状必须完全相同")

    q_heads, seq_len, head_dim = q.shape
    kv_heads, kv_seq_len, kv_head_dim = k.shape
    if head_dim <= 0 or head_dim % 2 or kv_head_dim != head_dim:
        raise ValueError("Q/K/V的head_dim必须相同且为正偶数")
    if seq_len <= 0 or kv_seq_len != seq_len:
        raise ValueError("Q/K/V的token形状必须相同且非空")
    if kv_heads <= 0 or q_heads <= 0 or q_heads % kv_heads:
        raise ValueError("Q头数必须能被KV头数整除")
    if sine.shape != (seq_len, head_dim // 2) or cosine.shape != sine.shape:
        raise ValueError("RoPE sin/cos形状与Q/K不匹配")
    if len(lut) != 513 or int(lut[0]) != 0x8000:
        raise ValueError("EXP LUT必须包含513项且首项为0x8000")
    if tile != 4:
        raise ValueError("当前RTL位感知参考只支持TILE=4")
    if seq_len % tile:
        raise ValueError("TILE=4必须整除token数")
    return q, k, v, sine, cosine


def rtl_aware_gqa_attention(
    q_words: np.ndarray,
    k_words: np.ndarray,
    v_words: np.ndarray,
    sine_words: np.ndarray,
    cosine_words: np.ndarray,
    lut: Sequence[int],
    *,
    tile: int = 4,
) -> np.ndarray:
    """镜像当前RTL舍入边界，返回``[Q头,token,dim]`` BF16位模式。"""

    q, k, v, sine, cosine = _validate_rtl_inputs(
        q_words, k_words, v_words, sine_words, cosine_words, lut, tile)
    q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[0]
    group_size = q_heads // kv_heads
    sine_flat = sine.reshape(-1).tolist()
    cosine_flat = cosine.reshape(-1).tolist()
    output = np.empty((q_heads, seq_len, head_dim), dtype=np.uint16)

    rotated_k: dict[tuple[int, int], list[int]] = {}
    for q_head in range(q_heads):
        kv_head = q_head // group_size
        values = [v[kv_head, column].tolist() for column in range(seq_len)]
        for row in range(seq_len):
            q_rotated = rope_vector(
                q[q_head, row].tolist(), row, sine_flat, cosine_flat)
            scores: list[int] = []
            masks: list[bool] = []
            for column in range(seq_len):
                masked = column > row
                masks.append(masked)
                if masked:
                    scores.append(0xFF80)
                    continue
                key = (kv_head, column)
                if key not in rotated_k:
                    rotated_k[key] = rope_vector(
                        k[kv_head, column].tolist(),
                        column,
                        sine_flat,
                        cosine_flat,
                    )
                scores.append(qk_score(q_rotated, rotated_k[key]))
            output[q_head, row] = np.asarray(
                flash_rtl_exact(scores, masks, values, lut, tile),
                dtype=np.uint16,
            )
    return output


__all__ = [
    "apply_split_half_rope",
    "bf16_words_to_float",
    "float_to_bf16_words",
    "rtl_aware_gqa_attention",
    "standard_gqa_attention",
]
