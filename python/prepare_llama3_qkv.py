#!/usr/bin/env python3
"""Generate layer-0, pre-RoPE Llama 3 Q/K/V for this board's DDR layout.

The board consumes complete 128-token BF16 arrays in [head, token, dim]
order. Short prompts are zero-padded after projection; their padded output
rows are not meaningful model tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "embedding": "model.embed_tokens.weight",
    "norm": "model.layers.0.input_layernorm.weight",
    "q": "model.layers.0.self_attn.q_proj.weight",
    "k": "model.layers.0.self_attn.k_proj.weight",
    "v": "model.layers.0.self_attn.v_proj.weight",
}
FILES = {
    "q": "q_before_rope_bf16.bin",
    "k": "k_before_rope_bf16.bin",
    "v": "v_bf16.bin",
}


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少 PyTorch，请在同一个 Python 环境安装 torch") from exc
    return torch


def weight_files(model_dir: Path) -> dict[str, Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        missing = [key for key in WEIGHTS.values() if key not in weight_map]
        if missing:
            raise ValueError(f"权重索引缺少: {', '.join(missing)}")
        paths = {key: model_dir / weight_map[key] for key in WEIGHTS.values()}
    else:
        single = model_dir / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError("需要 model.safetensors 或 model.safetensors.index.json 及对应分片")
        paths = {key: single for key in WEIGHTS.values()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少权重分片: {path}")
    return paths


def load_tensor(paths: dict[str, Path], key: str):
    from safetensors import safe_open

    with safe_open(str(paths[key]), framework="pt", device="cpu") as source:
        if key not in source.keys():
            raise ValueError(f"{paths[key]} 中缺少 {key}")
        return source.get_tensor(key)


def load_embedding_rows(paths: dict[str, Path], ids: list[int], hidden: int):
    """Read only the token rows needed; the full 8B embedding is about 1 GiB."""
    torch = require_torch()
    from safetensors import safe_open

    key = WEIGHTS["embedding"]
    with safe_open(str(paths[key]), framework="pt", device="cpu") as source:
        if key not in source.keys():
            raise ValueError(f"{paths[key]} 中缺少 {key}")
        embedding = source.get_slice(key)
        vocab, width = embedding.get_shape()
        if width != hidden:
            raise ValueError(f"Embedding 宽度应为 {hidden}，实际 {width}")
        if any(token_id < 0 or token_id >= vocab for token_id in ids):
            raise ValueError(f"Token ID 超出权重词表大小 {vocab}")
        rows = {token_id: embedding[token_id:token_id + 1].float()
                for token_id in set(ids)}
    return torch.cat([rows[token_id] for token_id in ids], dim=0)


def bf16_le_bytes(values: np.ndarray) -> bytes:
    """Round finite FP32 to BF16, ties to even, then serialize little endian."""
    f32 = np.ascontiguousarray(values, dtype="<f4")
    if not np.isfinite(f32).all():
        raise ValueError("Q/K/V 中出现 NaN 或无穷大")
    bits = f32.view("<u4")
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2").tobytes()


def project_layer0(model_dir: Path, ids: list[int], board: dict) -> dict[str, np.ndarray]:
    torch = require_torch()
    model_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if model_cfg.get("model_type") != "llama":
        raise ValueError("当前只支持 Hugging Face Llama 架构权重")
    if model_cfg.get("attention_bias", False):
        raise ValueError("当前板级导出未处理 Q/K/V 投影 bias")

    hidden = int(model_cfg["hidden_size"])
    q_heads = int(model_cfg["num_attention_heads"])
    kv_heads = int(model_cfg.get("num_key_value_heads", q_heads))
    head_dim = int(model_cfg.get("head_dim", hidden // q_heads))
    seq_len = int(board["seq_len"])
    if (q_heads, kv_heads, head_dim) != (int(board["q_heads"]), int(board["kv_heads"]), int(board["head_dim"])):
        raise ValueError("模型 Q/KV 头数或 head_dim 与当前 FPGA 工程不匹配")
    if hidden != q_heads * head_dim:
        raise ValueError("当前要求 hidden_size = num_attention_heads * head_dim")
    if not 1 <= len(ids) <= seq_len:
        raise ValueError(f"需要 1～{seq_len} 个 token（包含 BOS）；不会自动截断")

    paths = weight_files(model_dir)
    embeddings = load_embedding_rows(paths, ids, hidden)
    norm_weight = load_tensor(paths, WEIGHTS["norm"]).float()
    if tuple(norm_weight.shape) != (hidden,):
        raise ValueError(f"input_layernorm 权重形状错误: {tuple(norm_weight.shape)}")
    eps = float(model_cfg.get("rms_norm_eps", 1e-6))
    with torch.no_grad():
        x = embeddings.float()
        normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        normalized *= norm_weight
        del embeddings, norm_weight, x

        arrays = {}
        for name, heads in (("q", q_heads), ("k", kv_heads), ("v", kv_heads)):
            weight = load_tensor(paths, WEIGHTS[name]).float()
            expected = (heads * head_dim, hidden)
            if tuple(weight.shape) != expected:
                raise ValueError(f"{WEIGHTS[name]} 形状应为 {expected}，实际 {tuple(weight.shape)}")
            projected = torch.nn.functional.linear(normalized, weight)
            # HF Llama projection is [token, head, dim]; DDR is [head, token, dim].
            valid = projected.reshape(len(ids), heads, head_dim).permute(1, 0, 2)
            padded = torch.zeros((heads, seq_len, head_dim), dtype=torch.float32)
            padded[:, :len(ids), :] = valid
            arrays[name] = padded.numpy()
            del weight, projected, valid, padded
    return arrays


def encode_text(model_dir: Path, text: str) -> list[int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("缺少 transformers，请在同一个 Python 环境安装 transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True,
                                               local_files_only=True)
    return tokenizer.encode(text, add_special_tokens=True)


def write_outputs(output_dir: Path, arrays: dict[str, np.ndarray], ids: list[int], board: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_len = int(board["seq_len"])
    head_dim = int(board["head_dim"])
    files = {}
    for name in ("q", "k", "v"):
        heads = int(board["q_heads"] if name == "q" else board["kv_heads"])
        expected = (heads, seq_len, head_dim)
        if arrays[name].shape != expected:
            raise ValueError(f"{name} 应为 {expected}，实际 {arrays[name].shape}")
        data = bf16_le_bytes(arrays[name])
        path = output_dir / FILES[name]
        path.write_bytes(data)
        files[name] = {"file": path.name, "ddr_base": board[f"{name}_base"],
                       "bytes": len(data), "crc32": f"{zlib.crc32(data):08x}",
                       "shape": list(expected)}

    manifest = {
        "model_stage": "Llama layer 0: embedding -> input RMSNorm -> Q/K/V projections; no RoPE",
        "token_ids": ids,
        "valid_tokens": len(ids),
        "board_seq_len": seq_len,
        "padded_rows": "zero; rows at token index >= valid_tokens are not model output",
        "dtype": "BF16 round-to-nearest-even, little-endian uint16",
        "layout": "[head][token][dim] contiguous",
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="本地官方 Llama 3 8B Hugging Face 权重目录")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="输出二进制 Q/K/V 及 manifest 的新目录")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="输入句子；使用模型目录中的 tokenizer")
    source.add_argument("--token-ids", nargs="+", type=int,
                        help="已经分词的 ID（会原样使用，不自动加 BOS）")
    args = parser.parse_args(argv)

    board = json.loads((ROOT / "project_config.json").read_text(encoding="utf-8"))
    ids = encode_text(args.model_dir, args.text) if args.text is not None else args.token_ids
    arrays = project_layer0(args.model_dir, ids, board)
    write_outputs(args.output_dir, arrays, ids, board)
    print(f"已生成第 0 层未做 RoPE 的 Q/K/V：{len(ids)} 个有效 token，板级长度 {board['seq_len']}")
    for name in ("q", "k", "v"):
        print(f"{name.upper()}: {args.output_dir / FILES[name]}")
    print(f"布局和 DDR 地址: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
