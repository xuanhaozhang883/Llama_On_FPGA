#!/usr/bin/env python3
"""Tokenize one prompt in a tokenizer-only Python environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from .prepare_llama3_qkv import encode_text, model_metadata
else:
    from prepare_llama3_qkv import encode_text, model_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ids = encode_text(args.model_dir, args.text)
    payload = model_metadata(args.model_dir)
    payload["token_ids"] = ids
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Token IDs ({len(ids)}): {ids}")
    print(f"文件: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
