#!/usr/bin/env python3
"""Convert the official Llama 3 embedding tensor to a compact INT8 table.

This is a one-time, chunked conversion. It reads one safetensors shard and
writes a per-row symmetric INT8 matrix plus FP32 row scales. The source and
destination are intended to live outside the project source directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .prepare_llama3_qkv import WEIGHTS, model_metadata, validate_model_identity, weight_files
    from .safetensors_stream import SafeTensorReader
else:
    from prepare_llama3_qkv import WEIGHTS, model_metadata, validate_model_identity, weight_files
    from safetensors_stream import SafeTensorReader


def quantize_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One scale per vocabulary row; zero rows remain exactly zero."""
    values = np.ascontiguousarray(rows, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embedding 分片必须是有限的二维矩阵")
    maximum = np.max(np.abs(values), axis=1)
    scales = np.where(maximum == 0, np.float32(1.0), maximum / np.float32(127.0))
    quantized = np.rint(values / scales[:, None]).clip(-127, 127).astype(np.int8)
    return quantized, scales.astype(np.float32)


def quantize_embedding(model_dir: Path, output_dir: Path, chunk_rows: int = 256) -> dict:
    if chunk_rows < 1:
        raise ValueError("chunk_rows 必须为正数")
    model_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if model_cfg.get("model_type") != "llama":
        raise ValueError("当前只支持 Hugging Face Llama 架构权重")
    hidden = int(model_cfg["hidden_size"])
    key = WEIGHTS["embedding"]
    source_path = weight_files(model_dir, (key,))[key]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Do not leave an old completion record if an interrupted conversion rewrites arrays.
    manifest_path = output_dir / "quantization_manifest.json"
    manifest_path.unlink(missing_ok=True)
    with SafeTensorReader(source_path) as source:
        vocab, width = source.shape(key)
        if width != hidden:
            raise ValueError(f"Embedding 宽度应为 {hidden}，实际 {width}")
        matrix = np.lib.format.open_memmap(
            output_dir / "embedding_int8.npy", mode="w+", dtype=np.int8,
            shape=(vocab, hidden))
        scales = np.lib.format.open_memmap(
            output_dir / "embedding_scales_fp32.npy", mode="w+", dtype=np.float32,
            shape=(vocab,))
        for start in range(0, vocab, chunk_rows):
            end = min(start + chunk_rows, vocab)
            q, s = quantize_rows(source.read_row_range_fp32(key, start, end))
            matrix[start:end] = q
            scales[start:end] = s
        matrix.flush()
        scales.flush()
        del matrix, scales

    manifest = {
        **model_metadata(model_dir),
        "format": "symmetric INT8 per vocabulary row, FP32 scale; value ~= int8 * scale",
        "scope": "optional experiment; not the official BF16 layer-0 acceptance baseline",
        "shape": [vocab, hidden],
        "weights_file": "embedding_int8.npy",
        "scales_file": "embedding_scales_fp32.npy",
        "source_tensor": key,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_int8_rows(model_dir: Path, int8_dir: Path,
                   ids: list[int], hidden: int) -> np.ndarray:
    manifest = json.loads((int8_dir / "quantization_manifest.json").read_text(encoding="utf-8"))
    validate_model_identity(manifest, model_dir, "INT8 Embedding 表")
    matrix = np.load(int8_dir / "embedding_int8.npy", mmap_mode="r", allow_pickle=False)
    scales = np.load(int8_dir / "embedding_scales_fp32.npy", mmap_mode="r", allow_pickle=False)
    if (matrix.ndim != 2 or matrix.shape[1] != hidden or matrix.dtype != np.int8
            or scales.shape != (matrix.shape[0],) or scales.dtype != np.float32
            or manifest.get("shape") != list(matrix.shape)):
        raise ValueError("INT8 Embedding 表尺寸或类型错误")
    if any(type(token_id) is not int or token_id < 0 or token_id >= matrix.shape[0]
           for token_id in ids):
        raise ValueError("Token ID 超出 INT8 Embedding 词表范围")
    selected_scales = np.asarray(scales[ids], dtype=np.float32)
    if not np.isfinite(selected_scales).all() or np.any(selected_scales <= 0):
        raise ValueError("INT8 Embedding scale 无效")
    return np.asarray(matrix[ids], dtype=np.float32) * selected_scales[:, None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="官方 BF16 模型目录，包含索引和 Embedding 分片")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="INT8 表输出目录；建议放在权重目录内，不放源码内")
    parser.add_argument("--chunk-rows", type=int, default=256,
                        help="每次转换的词表行数，默认 256")
    args = parser.parse_args(argv)
    manifest = quantize_embedding(args.model_dir, args.output_dir, args.chunk_rows)
    vocab, hidden = manifest["shape"]
    print(f"INT8 Embedding 已生成: {vocab} x {hidden}")
    print(f"目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
