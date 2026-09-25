#!/usr/bin/env python3
"""Tokenize a prompt and save Llama 3 layer-0 token embeddings on the PC.

Only the embedding tensor's shard is needed. Model weights are read from
--model-dir and are never copied into the source tree or output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__:
    from .prepare_llama3_qkv import (RESIDUAL_FILE, ROOT, WEIGHTS, model_metadata,
                                     encode_text, load_token_ids_file, weight_files)
    from .quantize_llama3_embedding import load_int8_rows
    from .safetensors_stream import SafeTensorReader
else:
    from prepare_llama3_qkv import (RESIDUAL_FILE, ROOT, WEIGHTS, model_metadata,
                                    encode_text, load_token_ids_file, weight_files)
    from quantize_llama3_embedding import load_int8_rows
    from safetensors_stream import SafeTensorReader


def prepare_embedding(model_dir: Path, ids: list[int], board: dict,
                      int8_dir: Path | None = None) -> np.ndarray:
    model_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if model_cfg.get("model_type") != "llama":
        raise ValueError("当前只支持 Hugging Face Llama 架构权重")
    hidden = int(model_cfg["hidden_size"])
    expected_hidden = int(board["q_heads"]) * int(board["head_dim"])
    if hidden != expected_hidden:
        raise ValueError(f"模型 hidden_size={hidden}，板级工程要求 {expected_hidden}")
    if not 1 <= len(ids) <= int(board["seq_len"]):
        raise ValueError(f"需要 1～{board['seq_len']} 个 token（包含 BOS）；不会自动截断")
    if int8_dir is not None:
        return load_int8_rows(model_dir, int8_dir, ids, hidden)
    key = WEIGHTS["embedding"]
    path = weight_files(model_dir, (key,))[key]
    with SafeTensorReader(path) as source:
        shape = source.shape(key)
        if len(shape) != 2 or shape[1] != hidden:
            raise ValueError(f"Embedding 权重形状应为 [vocab, {hidden}]，实际 {shape}")
        return source.read_rows_fp32(key, ids)


def save_embedding(output_dir: Path, model_dir: Path,
                   ids: list[int], embedding: np.ndarray,
                   weight_format: str = "safetensors") -> Path:
    if embedding.ndim != 2 or embedding.shape[0] != len(ids) or embedding.dtype != np.float32:
        raise ValueError("Embedding 应为 [token, hidden_size] FP32 数组")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding 中出现 NaN 或无穷大")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "embedding_manifest.json"
    manifest_path.unlink(missing_ok=True)
    target = output_dir / "embedding_fp32.npy"
    np.save(target, np.ascontiguousarray(embedding, dtype="<f4"), allow_pickle=False)
    payload_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    residual_path = output_dir / RESIDUAL_FILE
    np.save(residual_path, np.ascontiguousarray(embedding[None, :, :], dtype="<f4"), allow_pickle=False)
    manifest = {
        "stage": "Llama layer 0 token embedding, before RMSNorm and Q/K/V projection",
        **model_metadata(model_dir),
        "token_ids": ids,
        "valid_tokens": len(ids),
        "shape": list(embedding.shape),
        "dtype": "float32",
        "source_weight_format": weight_format,
        "file": target.name,
        "sha256": payload_hash,
        "residual_hidden": {
            "file": residual_path.name, "shape": [1, *embedding.shape],
            "dtype": "float32 little-endian", "layout": "[batch][token][hidden]",
            "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
            "stage": "same token embedding before input RMSNorm; unpadded",
        },
    }
    manifest_temp = output_dir / "embedding_manifest.json.tmp"
    manifest_temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_temp.replace(manifest_path)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="源码目录外的 Hugging Face Llama 3 8B 权重目录")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="保存 Embedding 数组和 manifest 的目录")
    parser.add_argument("--int8-dir", type=Path,
                        help="可选：读取此前转换的 INT8 Embedding 表")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="输入句子，自动分词并添加 BOS")
    source.add_argument("--token-ids", nargs="+", type=int,
                        help="已有的 token IDs；不会再添加 BOS")
    source.add_argument("--token-ids-file", type=Path,
                        help="从 tokenize_llama3.py 生成的 JSON 读取 token IDs")
    args = parser.parse_args(argv)

    board = json.loads((ROOT / "project_config.json").read_text(encoding="utf-8"))
    start = time.perf_counter()
    if args.text is not None:
        ids = encode_text(args.model_dir, args.text)
    elif args.token_ids_file is not None:
        ids = load_token_ids_file(args.token_ids_file, args.model_dir)
    else:
        ids = args.token_ids
    token_seconds = time.perf_counter() - start
    start = time.perf_counter()
    embedding = prepare_embedding(args.model_dir, ids, board, args.int8_dir)
    embedding_seconds = time.perf_counter() - start
    weight_format = "symmetric INT8 per row" if args.int8_dir else "safetensors"
    target = save_embedding(args.output_dir, args.model_dir, ids, embedding, weight_format)
    print(f"Token IDs ({len(ids)}): {ids}")
    print(f"Embedding: {embedding.shape}, FP32, {embedding.nbytes} bytes")
    print(f"文件: {target}")
    print(f"分词耗时: {token_seconds:.4f} s；Embedding 读取耗时: {embedding_seconds:.4f} s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
