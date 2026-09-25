#!/usr/bin/env python3
"""Check saved embeddings against the selected weight rows and metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .prepare_llama3_qkv import WEIGHTS, validate_model_identity, weight_files
    from .quantize_llama3_embedding import load_int8_rows
    from .safetensors_stream import SafeTensorReader
else:
    from prepare_llama3_qkv import WEIGHTS, validate_model_identity, weight_files
    from quantize_llama3_embedding import load_int8_rows
    from safetensors_stream import SafeTensorReader


def verify_embedding(model_dir: Path, embedding_dir: Path,
                     int8_dir: Path | None = None) -> tuple[int, int]:
    manifest = json.loads((embedding_dir / "embedding_manifest.json").read_text(encoding="utf-8"))
    validate_model_identity(manifest, model_dir, "Embedding 输出")
    ids = manifest.get("token_ids")
    if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
        raise ValueError("Embedding 输出的 token IDs 无效")
    path = embedding_dir / "embedding_fp32.npy"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("Embedding 输出文件 SHA256 校验失败")
    actual = np.load(path, allow_pickle=False)
    hidden = int(json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["hidden_size"])
    if actual.dtype != np.float32 or actual.shape != (len(ids), hidden):
        raise ValueError(f"Embedding 输出形状或类型错误: {actual.shape}, {actual.dtype}")
    if not np.isfinite(actual).all():
        raise ValueError("Embedding 输出包含非有限数值")
    if manifest.get("shape") != [len(ids), hidden]:
        raise ValueError("Embedding manifest 形状不匹配")

    if int8_dir is not None:
        expected = load_int8_rows(model_dir, int8_dir, ids, hidden)
    else:
        key = WEIGHTS["embedding"]
        path = weight_files(model_dir, (key,))[key]
        with SafeTensorReader(path) as source:
            if source.shape(key)[1] != hidden:
                raise ValueError("权重 Embedding 宽度与配置不匹配")
            expected = source.read_rows_fp32(key, ids)
    if not np.array_equal(actual, expected):
        error = float(np.max(np.abs(actual - expected)))
        raise ValueError(f"Embedding 与权重行不一致，最大绝对误差 {error}")
    return len(ids), hidden


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--int8-dir", type=Path,
                        help="量化 Embedding 输出验证时指定同一个 INT8 表目录")
    args = parser.parse_args(argv)
    tokens, hidden = verify_embedding(args.model_dir, args.embedding_dir, args.int8_dir)
    print(f"[PASS] 每个 token 的 Embedding 都等于权重表中对应 ID 的行：{tokens} 个 token，每个 {hidden} 维")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
