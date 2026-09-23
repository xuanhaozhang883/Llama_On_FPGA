"""Llama decoder第0层Attention之后的CPU FP32计算。

FPGA返回的Context为BF16小端 ``[head][token][dim]``。本模块负责
拼头、选择性读取后半层权重，以及执行 ``o_proj + residual + MLP``。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from python.prepare_llama3_qkv import load_tensor, weight_files


DEFAULT_Q_HEADS = 32
DEFAULT_SEQ_LEN = 128
DEFAULT_HEAD_DIM = 128


@dataclass(frozen=True)
class PostWeights:
    o_proj: torch.Tensor
    post_attention_layernorm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


@dataclass(frozen=True)
class Layer0PostResult:
    context_merged: torch.Tensor
    o_proj: torch.Tensor
    residual_1: torch.Tensor
    post_norm: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    swiglu: torch.Tensor
    down_proj: torch.Tensor
    layer_output: torch.Tensor


def decode_context_bf16(
    payload: bytes | bytearray | memoryview,
    *,
    valid_tokens: int,
    q_heads: int = DEFAULT_Q_HEADS,
    seq_len: int = DEFAULT_SEQ_LEN,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> torch.Tensor:
    """Decode BF16 LE ``[head, token, dim]`` into FP32 ``[1, L, hidden]``."""
    dimensions = (q_heads, seq_len, head_dim)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("q_heads、seq_len和head_dim必须是正整数")
    if type(valid_tokens) is not int or not 1 <= valid_tokens <= seq_len:
        raise ValueError(f"valid_tokens必须在1～{seq_len}之间")

    expected_bytes = q_heads * seq_len * head_dim * 2
    if len(payload) != expected_bytes:
        raise ValueError(f"Context载荷必须恰好为{expected_bytes}字节，实际{len(payload)}字节")

    words = np.frombuffer(payload, dtype="<u2").astype("<u4")
    values = (words << np.uint32(16)).view("<f4")
    heads = values.reshape(q_heads, seq_len, head_dim)
    merged = heads.transpose(1, 0, 2).reshape(1, seq_len, q_heads * head_dim)
    merged = np.ascontiguousarray(merged[:, :valid_tokens, :], dtype=np.float32)
    if not np.isfinite(merged).all():
        raise ValueError("Context中出现NaN或无穷大")
    return torch.from_numpy(merged)


def _weight_names(layer_idx: int) -> dict[str, str]:
    prefix = f"model.layers.{layer_idx}"
    return {
        "o_proj": f"{prefix}.self_attn.o_proj.weight",
        "post_attention_layernorm": f"{prefix}.post_attention_layernorm.weight",
        "gate_proj": f"{prefix}.mlp.gate_proj.weight",
        "up_proj": f"{prefix}.mlp.up_proj.weight",
        "down_proj": f"{prefix}.mlp.down_proj.weight",
    }


def _validate_weight_shapes(weights: PostWeights, hidden: int | None = None,
                            intermediate: int | None = None) -> tuple[int, int]:
    tensors = {
        "o_proj": weights.o_proj,
        "post_attention_layernorm": weights.post_attention_layernorm,
        "gate_proj": weights.gate_proj,
        "up_proj": weights.up_proj,
        "down_proj": weights.down_proj,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name}必须是torch.Tensor")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"{name}必须是CPU FP32")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}中出现NaN或无穷大")

    if weights.o_proj.ndim != 2 or weights.o_proj.shape[0] != weights.o_proj.shape[1]:
        raise ValueError("o_proj形状必须为[hidden, hidden]")
    actual_hidden = int(weights.o_proj.shape[0])
    if weights.post_attention_layernorm.shape != (actual_hidden,):
        raise ValueError("post_attention_layernorm形状必须为[hidden]")
    if weights.gate_proj.ndim != 2 or weights.gate_proj.shape[1] != actual_hidden:
        raise ValueError("gate_proj形状必须为[intermediate, hidden]")
    actual_intermediate = int(weights.gate_proj.shape[0])
    if weights.up_proj.shape != (actual_intermediate, actual_hidden):
        raise ValueError("up_proj形状必须与gate_proj一致")
    if weights.down_proj.shape != (actual_hidden, actual_intermediate):
        raise ValueError("down_proj形状必须为[hidden, intermediate]")
    if hidden is not None and actual_hidden != hidden:
        raise ValueError(f"后半层hidden_size应为{hidden}，实际{actual_hidden}")
    if intermediate is not None and actual_intermediate != intermediate:
        raise ValueError(
            f"后半层intermediate_size应为{intermediate}，实际{actual_intermediate}")
    return actual_hidden, actual_intermediate


def load_post_weights(model_dir: Path, layer_idx: int = 0) -> PostWeights:
    """Selectively read only the five tensors needed after Attention."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "llama":
        raise ValueError("当前只支持Hugging Face Llama架构权重")
    layers = int(config.get("num_hidden_layers", 0))
    if type(layer_idx) is not int or not 0 <= layer_idx < layers:
        raise ValueError(f"layer_idx必须在0～{layers - 1}之间")
    if config.get("attention_bias", False) or config.get("mlp_bias", False):
        raise ValueError("当前后半层实现不支持Attention或MLP bias")

    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    names = _weight_names(layer_idx)
    paths = weight_files(model_dir, tuple(names.values()))
    weights = PostWeights(**{
        field: load_tensor(paths, tensor_name).to(device="cpu", dtype=torch.float32).contiguous()
        for field, tensor_name in names.items()
    })
    _validate_weight_shapes(weights, hidden, intermediate)
    return weights


def run_post_attention(
    context: torch.Tensor,
    residual_hidden: torch.Tensor,
    weights: PostWeights,
    *,
    rms_eps: float,
) -> Layer0PostResult:
    """Run Llama ``o_proj -> residual -> RMSNorm -> SwiGLU -> residual``."""
    if not isinstance(context, torch.Tensor) or not isinstance(residual_hidden, torch.Tensor):
        raise ValueError("Context和Residual必须是torch.Tensor")
    for name, tensor in (("Context", context), ("Residual", residual_hidden)):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"{name}必须是CPU FP32")
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] < 1:
            raise ValueError(f"{name}维度必须为[1,L,hidden]且L不小于1")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}中出现NaN或无穷大")
    if context.shape != residual_hidden.shape:
        raise ValueError("Context和Residual形状必须一致")
    if not isinstance(rms_eps, (int, float)) or not math.isfinite(rms_eps) or rms_eps <= 0:
        raise ValueError("rms_eps必须是有限正数")

    hidden, _ = _validate_weight_shapes(weights)
    if context.shape[-1] != hidden:
        raise ValueError(f"Context最后一维应为{hidden}，实际{context.shape[-1]}")

    with torch.no_grad():
        context_merged = context.contiguous().clone()
        o_proj = functional.linear(context_merged, weights.o_proj)
        residual_1 = o_proj + residual_hidden
        variance = residual_1.square().mean(dim=-1, keepdim=True)
        post_norm = residual_1 * torch.rsqrt(variance + float(rms_eps))
        post_norm = post_norm * weights.post_attention_layernorm
        gate = functional.linear(post_norm, weights.gate_proj)
        up = functional.linear(post_norm, weights.up_proj)
        swiglu = functional.silu(gate) * up
        down_proj = functional.linear(swiglu, weights.down_proj)
        layer_output = residual_1 + down_proj

    return Layer0PostResult(
        context_merged=context_merged,
        o_proj=o_proj,
        residual_1=residual_1,
        post_norm=post_norm,
        gate=gate,
        up=up,
        swiglu=swiglu,
        down_proj=down_proj,
        layer_output=layer_output,
    )
