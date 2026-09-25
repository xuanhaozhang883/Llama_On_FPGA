#!/usr/bin/env python3
"""Generate layer-0, pre-RoPE Llama 3 Q/K/V for this board's DDR layout.

The board consumes complete 128-token BF16 arrays in [head, token, dim]
order. Short prompts are zero-padded after projection; their padded output
rows are not meaningful model tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

if __package__:
    from .safetensors_stream import SafeTensorReader
else:
    from safetensors_stream import SafeTensorReader

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
RESIDUAL_FILE = "residual_hidden_fp32.npy"


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少 PyTorch，请在同一个 Python 环境安装 torch") from exc
    return torch


def weight_files(model_dir: Path, required_keys=None) -> dict[str, Path]:
    """Resolve only the tensors needed for this stage of the pipeline."""
    if required_keys is None:
        required_keys = tuple(WEIGHTS.values())
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        missing = [key for key in required_keys if key not in weight_map]
        if missing:
            raise ValueError(f"权重索引缺少: {', '.join(missing)}")
        paths = {key: model_dir / weight_map[key] for key in required_keys}
    else:
        single = model_dir / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError("需要 model.safetensors 或 model.safetensors.index.json 及对应分片")
        paths = {key: single for key in required_keys}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少权重分片: {path}")
    return paths


def load_tensor(paths: dict[str, Path], key: str):
    torch = require_torch()
    with SafeTensorReader(paths[key]) as source:
        return torch.from_numpy(source.read_tensor_fp32(key))


def load_embedding_rows(paths: dict[str, Path], ids: list[int], hidden: int):
    """Read selected rows without mapping the multi-gigabyte shard."""
    torch = require_torch()
    key = WEIGHTS["embedding"]
    with SafeTensorReader(paths[key]) as source:
        vocab, width = source.shape(key)
        if width != hidden:
            raise ValueError(f"Embedding 宽度应为 {hidden}，实际 {width}")
        return torch.from_numpy(source.read_rows_fp32(key, ids))


def config_sha256(model_dir: Path) -> str:
    return hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest()


def model_metadata(model_dir: Path) -> dict:
    """Identify the files actually used; never infer a model name from dimensions."""
    config_path = model_dir / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    index = model_dir / "model.safetensors.index.json"
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    model_config_fields = (
        "_name_or_path", "model_type", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "vocab_size", "rms_norm_eps", "max_position_embeddings",
        "torch_dtype", "dtype", "rope_theta", "rope_scaling", "rope_parameters",
    )
    metadata = {
        "model_dir": str(model_dir.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weight_index_file": index.name if index.is_file() else None,
        "weight_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None,
        "tokenizer_sha256": {
            name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
            for name in tokenizer_files if (model_dir / name).is_file()
        },
        "model_config": {key: config.get(key) for key in model_config_fields},
        "model_bundle_sha256": None,
        "model_bundle": None,
    }
    bundle_path = model_dir / "model-bundle-manifest.json"
    if bundle_path.is_file():
        bundle_bytes = bundle_path.read_bytes()
        bundle = json.loads(bundle_bytes)
        if bundle.get("schema_version") != 1 or bundle.get("status") != "complete":
            raise ValueError("模型 bundle 清单不是完整且受支持的版本")
        records = bundle.get("files", {})
        small_hashes = {"config.json": metadata["config_sha256"], **metadata["tokenizer_sha256"]}
        if metadata["weight_index_sha256"] is not None:
            small_hashes[index.name] = metadata["weight_index_sha256"]
        for name, digest in small_hashes.items():
            if records.get(name, {}).get("sha256") != digest:
                raise ValueError(f"模型 bundle 清单与当前 {name} 的 SHA256 不匹配")
        # Also reject a removed tokenizer/index declared by the completed bundle.
        for name in (*tokenizer_files, index.name):
            if name in records and name not in small_hashes:
                raise ValueError(f"模型 bundle 清单中的文件缺失: {name}")
        metadata["model_bundle_sha256"] = hashlib.sha256(bundle_bytes).hexdigest()
        metadata["model_bundle"] = {
            "file": bundle_path.name, "repo_id": bundle.get("repo_id"),
            "revision": bundle.get("revision"), "source": bundle.get("source"),
            "required_weight_shards": bundle.get("required_weight_shards", []),
            "weight_verification": "recorded by bundle preparation; not rehashed during QKV/embedding export",
        }
    return metadata


def validate_model_identity(metadata: dict, model_dir: Path, label: str) -> None:
    """Paths are provenance only. Preserve strict hashes, including legacy config checks."""
    actual = model_metadata(model_dir)
    if metadata.get("config_sha256") != actual["config_sha256"]:
        raise ValueError(f"{label}与当前模型配置不匹配")
    for key, description in (("weight_index_sha256", "权重索引"),
                             ("tokenizer_sha256", "Tokenizer"),
                             ("model_bundle_sha256", "bundle 身份")):
        if key in metadata and metadata[key] != actual[key]:
            raise ValueError(f"{label}与当前模型{description}不匹配")


def load_token_ids_file(path: Path, model_dir: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_model_identity(data, model_dir, "Token IDs 文件")
    ids = data.get("token_ids")
    if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
        raise ValueError("Token IDs 文件内容无效")
    return ids


def load_saved_embedding(embedding_dir: Path, model_dir: Path,
                         ids: list[int], hidden: int):
    """Reject stale or mismatched embedding output before QKV projection."""
    torch = require_torch()
    manifest = json.loads((embedding_dir / "embedding_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("token_ids") != ids:
        raise ValueError("Embedding 文件的 token IDs 与本次输入不同")
    validate_model_identity(manifest, model_dir, "Embedding 文件")
    path = embedding_dir / "embedding_fp32.npy"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("Embedding 文件 SHA256 校验失败")
    array = np.load(path, allow_pickle=False)
    if array.shape != (len(ids), hidden) or array.dtype != np.float32:
        raise ValueError(f"Embedding 文件形状或类型错误: {array.shape}, {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("Embedding 文件含 NaN 或无穷大")
    return torch.from_numpy(np.ascontiguousarray(array))


def bf16_le_bytes(values: np.ndarray) -> bytes:
    """Round finite FP32 to BF16, ties to even, then serialize little endian."""
    f32 = np.ascontiguousarray(values, dtype="<f4")
    if not np.isfinite(f32).all():
        raise ValueError("Q/K/V 中出现 NaN 或无穷大")
    bits = f32.view("<u4")
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2").tobytes()


def resolve_compute(device: str, compute_dtype: str) -> tuple[str, str]:
    torch = require_torch()
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device 必须为 auto、cpu 或 cuda")
    if compute_dtype not in ("auto", "fp32", "bf16"):
        raise ValueError("compute_dtype 必须为 auto、fp32 或 bf16")
    actual_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if actual_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA；请安装 CUDA 版 PyTorch 并检查驱动")
    actual_dtype = ("bf16" if actual_device == "cuda" else "fp32") if compute_dtype == "auto" else compute_dtype
    if actual_device == "cuda" and actual_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16，请选择 --compute-dtype fp32")
    return actual_device, actual_dtype


def project_layer0(model_dir: Path, ids: list[int], board: dict,
                   embeddings=None, device: str = "cpu",
                   compute_dtype: str = "fp32") -> dict[str, np.ndarray]:
    """Return padded Q/K/V and the unpadded, pre-RMSNorm residual [1,L,H]."""
    torch = require_torch()
    actual_device, actual_dtype = resolve_compute(device, compute_dtype)
    torch_dtype = torch.bfloat16 if actual_dtype == "bf16" else torch.float32
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
    if any(type(token_id) is not int or token_id < 0
           or ("vocab_size" in model_cfg and token_id >= int(model_cfg["vocab_size"]))
           for token_id in ids):
        raise ValueError("Token ID 超出模型词表范围或不是整数")

    names = ("norm", "q", "k", "v")
    if embeddings is None:
        names = ("embedding",) + names
    required = tuple(WEIGHTS[name] for name in names)
    paths = weight_files(model_dir, required)
    if embeddings is None:
        embeddings = load_embedding_rows(paths, ids, hidden)
    elif tuple(embeddings.shape) != (len(ids), hidden):
        raise ValueError(f"Embedding 应为 {(len(ids), hidden)}，实际 {tuple(embeddings.shape)}")
    norm_weight = load_tensor(paths, WEIGHTS["norm"]).float()
    if tuple(norm_weight.shape) != (hidden,):
        raise ValueError(f"input_layernorm 权重形状错误: {tuple(norm_weight.shape)}")
    eps = float(model_cfg.get("rms_norm_eps", 1e-6))
    with torch.no_grad():
        x = torch.as_tensor(embeddings).detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError("Embedding 文件含 NaN 或无穷大")
        residual = x.unsqueeze(0).contiguous().clone().numpy()
        normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        normalized *= norm_weight
        normalized = normalized.to(device=actual_device, dtype=torch_dtype)
        del embeddings, norm_weight, x

        arrays = {"residual_hidden": residual}
        for name, heads in (("q", q_heads), ("k", kv_heads), ("v", kv_heads)):
            weight = load_tensor(paths, WEIGHTS[name])
            expected = (heads * head_dim, hidden)
            if tuple(weight.shape) != expected:
                raise ValueError(f"{WEIGHTS[name]} 形状应为 {expected}，实际 {tuple(weight.shape)}")
            weight = weight.to(device=actual_device, dtype=torch_dtype)
            projected = torch.nn.functional.linear(normalized, weight)
            # HF Llama projection is [token, head, dim]; DDR is [head, token, dim].
            valid = projected.reshape(len(ids), heads, head_dim).permute(1, 0, 2)
            valid = valid.to(device="cpu", dtype=torch.float32)
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


def write_outputs(output_dir: Path, arrays: dict[str, np.ndarray], ids: list[int],
                  board: dict, compute: dict | None = None,
                  model_dir: Path | None = None, embedding_source: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    seq_len = int(board["seq_len"])
    head_dim = int(board["head_dim"])
    if not 1 <= len(ids) <= seq_len:
        raise ValueError(f"需要 1～{seq_len} 个有效 token")
    files = {}
    for name in ("q", "k", "v"):
        heads = int(board["q_heads"] if name == "q" else board["kv_heads"])
        expected = (heads, seq_len, head_dim)
        if arrays[name].shape != expected:
            raise ValueError(f"{name} 应为 {expected}，实际 {arrays[name].shape}")
        if np.any(arrays[name][:, len(ids):, :] != 0):
            raise ValueError(f"{name} 的无效 token 行必须补零")
        data = bf16_le_bytes(arrays[name])
        path = output_dir / FILES[name]
        path.write_bytes(data)
        files[name] = {"file": path.name, "ddr_base": board[f"{name}_base"],
                       "bytes": len(data), "crc32": f"{zlib.crc32(data):08x}",
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "shape": list(expected)}

    if "residual_hidden" in arrays:
        residual = arrays["residual_hidden"]
        expected = (1, len(ids), int(board["q_heads"]) * head_dim)
        if residual.shape != expected or residual.dtype != np.float32:
            raise ValueError(f"Residual 应为 {expected} FP32，实际 {residual.shape}, {residual.dtype}")
        if not np.isfinite(residual).all():
            raise ValueError("Residual 中出现 NaN 或无穷大")
        path = output_dir / RESIDUAL_FILE
        np.save(path, np.ascontiguousarray(residual, dtype="<f4"), allow_pickle=False)
        data = path.read_bytes()
        files["residual_hidden"] = {
            "file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "shape": list(expected), "dtype": "float32 little-endian", "layout": "[batch][token][hidden]",
            "stage": "token embedding before input RMSNorm; unpadded; PC-side residual only",
        }

    manifest = {
        "model_stage": "Llama layer 0: embedding -> input RMSNorm -> Q/K/V projections; no RoPE",
        "manifest_version": 2,
        "token_ids": ids,
        "valid_tokens": len(ids),
        "board_seq_len": seq_len,
        "padded_rows": "zero; rows at token index >= valid_tokens are not model output",
        "dtype": "BF16 round-to-nearest-even, little-endian uint16",
        "layout": "[head][token][dim] contiguous",
        "files": files,
    }
    if compute is not None:
        manifest["host_compute"] = compute
    if model_dir is not None:
        manifest.update(model_metadata(model_dir))
    if embedding_source is not None:
        manifest["embedding_source"] = embedding_source
    manifest_temp = output_dir / "manifest.json.tmp"
    manifest_temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_temp.replace(manifest_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="本地官方 Llama 3 8B Hugging Face 权重目录")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="输出二进制 Q/K/V 及 manifest 的新目录")
    parser.add_argument("--embedding-dir", type=Path,
                        help="可选：读取第一步生成的 embedding_fp32.npy，跳过 Embedding 权重读取")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu",
                        help="矩阵乘设备；联调默认 cpu；auto 优先 CUDA")
    parser.add_argument("--compute-dtype", choices=("auto", "fp32", "bf16"), default="fp32",
                        help="联调默认 fp32；auto 在 CUDA 上用 BF16、CPU 上用 FP32")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="PyTorch CPU 线程数，联调默认 1；跨机器仍需核对软件版本")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="输入句子；使用模型目录中的 tokenizer")
    source.add_argument("--token-ids", nargs="+", type=int,
                        help="已经分词的 ID（会原样使用，不自动加 BOS）")
    source.add_argument("--token-ids-file", type=Path,
                        help="从 tokenize_llama3.py 生成的 JSON 读取 token IDs")
    args = parser.parse_args(argv)
    if args.num_threads < 1:
        parser.error("--num-threads 必须大于 0")
    torch = require_torch()
    torch.set_num_threads(args.num_threads)

    board = json.loads((ROOT / "project_config.json").read_text(encoding="utf-8"))
    if args.text is not None:
        ids = encode_text(args.model_dir, args.text)
    elif args.token_ids_file is not None:
        ids = load_token_ids_file(args.token_ids_file, args.model_dir)
    else:
        ids = args.token_ids
    embeddings = None
    embedding_source = {"source_weight_format": "safetensors", "load_dtype": "float32"}
    if args.embedding_dir is not None:
        hidden = int(board["q_heads"]) * int(board["head_dim"])
        embeddings = load_saved_embedding(args.embedding_dir, args.model_dir, ids, hidden)
        embedding_manifest = args.embedding_dir / "embedding_manifest.json"
        saved = json.loads(embedding_manifest.read_text(encoding="utf-8"))
        embedding_source = {
            "source_weight_format": saved.get("source_weight_format", "unspecified"),
            "embedding_manifest_sha256": hashlib.sha256(embedding_manifest.read_bytes()).hexdigest(),
            "embedding_sha256": saved["sha256"], "load_dtype": "float32",
        }
    actual_device, actual_dtype = resolve_compute(args.device, args.compute_dtype)
    if actual_device == "cpu":
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    arrays = project_layer0(args.model_dir, ids, board, embeddings,
                            actual_device, actual_dtype)
    elapsed = time.perf_counter() - start
    write_outputs(args.output_dir, arrays, ids, board,
                  {"device": actual_device, "dtype": actual_dtype,
                   "rmsnorm_device": "cpu", "rmsnorm_dtype": "fp32",
                   "projection_input_dtype": actual_dtype, "projection_weight_dtype": actual_dtype,
                   "projection_output_dtype": actual_dtype, "residual_dtype": "fp32",
                   "torch_version": torch.__version__, "numpy_version": np.__version__,
                   "cpu_threads": torch.get_num_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if actual_device == "cuda" else None,
                   "preparation_seconds": elapsed},
                  model_dir=args.model_dir, embedding_source=embedding_source)
    print(f"已生成第 0 层未做 RoPE 的 Q/K/V：{len(ids)} 个有效 token，板级长度 {board['seq_len']}")
    print(f"矩阵乘设备/精度: {actual_device}/{actual_dtype}；权重读取及生成耗时: {elapsed:.4f} s")
    for name in ("q", "k", "v"):
        print(f"{name.upper()}: {args.output_dir / FILES[name]}")
    print(f"Residual（RMSNorm 前）: {args.output_dir / RESIDUAL_FILE}")
    print(f"布局和 DDR 地址: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
