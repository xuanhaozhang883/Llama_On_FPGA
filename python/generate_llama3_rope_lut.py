#!/usr/bin/env python3
"""Generate auditable Meta-Llama-3-8B RoPE candidate ROM tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from python.prepare_llama3_model import BundleError, REPO_ID, validate_config


SIN_FILE = "sin_bf16.hex"
COS_FILE = "cos_bf16.hex"
MANIFEST_FILE = "manifest.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_words(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) >> np.uint32(16)).astype(np.uint16)


def generate_rope_words(
    config: Mapping[str, object],
    seq_len: int = 128,
    head_dim: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return token-major ``[position, frequency]`` BF16 bit patterns."""
    validate_config(config)
    if type(seq_len) is not int or not 1 <= seq_len <= int(config["max_position_embeddings"]):
        raise ValueError("seq_len必须是模型上下文范围内的正整数")
    if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim必须是正偶数")
    positions = torch.arange(seq_len, dtype=torch.float32)
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (float(config["rope_theta"]) ** (dimensions / head_dim))
    angles = torch.outer(positions, inv_freq)
    sine = angles.sin().to(torch.bfloat16).view(torch.uint16).numpy().copy()
    cosine = angles.cos().to(torch.bfloat16).view(torch.uint16).numpy().copy()
    return sine, cosine


def _hex_payload(words: np.ndarray) -> bytes:
    if words.dtype != np.uint16 or words.ndim != 2:
        raise ValueError("RoPE表必须是二维uint16数组")
    return "".join(f"{int(word):04X}\n" for word in words.reshape(-1)).encode("ascii")


def _write_bundle_files(output_dir: Path, payloads: dict[str, bytes]) -> None:
    temporary_paths = []
    try:
        for name, payload in payloads.items():
            temporary = output_dir / f"{name}.tmp"
            temporary.write_bytes(payload)
            temporary_paths.append(temporary)
        for name in payloads:
            (output_dir / f"{name}.tmp").replace(output_dir / name)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def write_lut_bundle(
    config_path: Path,
    output_dir: Path,
    forbidden_rom_dir: Path,
) -> Path:
    """Write candidate LUTs outside the active ROM directory."""
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    forbidden_rom_dir = Path(forbidden_rom_dir).resolve()
    if output_dir == forbidden_rom_dir or forbidden_rom_dir in output_dir.parents:
        raise ValueError(f"候选LUT不能写入活动mem目录: {forbidden_rom_dir}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise BundleError("config.json必须是JSON对象")
    sine, cosine = generate_rope_words(config)
    sin_payload = _hex_payload(sine)
    cos_payload = _hex_payload(cosine)
    manifest = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "config_file": config_path.name,
        "config_sha256": _sha256_bytes(config_bytes),
        "rope_theta": float(config["rope_theta"]),
        "max_position_embeddings": int(config["max_position_embeddings"]),
        "seq_len": int(sine.shape[0]),
        "head_dim": int(sine.shape[1] * 2),
        "frequency_count": int(sine.shape[1]),
        "element_count_per_table": int(sine.size),
        "algorithm": "torch_fp32_then_bf16_rne",
        "files": {
            "sin": {"file": SIN_FILE, "size_bytes": len(sin_payload),
                    "sha256": _sha256_bytes(sin_payload)},
            "cos": {"file": COS_FILE, "size_bytes": len(cos_payload),
                    "sha256": _sha256_bytes(cos_payload)},
        },
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_bundle_files(output_dir, {
        SIN_FILE: sin_payload,
        COS_FILE: cos_payload,
        MANIFEST_FILE: manifest_payload,
    })
    return output_dir / MANIFEST_FILE


def _read_hex_words(path: Path, shape: tuple[int, int]) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    expected = shape[0] * shape[1]
    if len(lines) != expected:
        raise ValueError(f"{path}应包含{expected}项，实际{len(lines)}项")
    if any(re.fullmatch(r"[0-9A-Fa-f]{4}", line) is None for line in lines):
        raise ValueError(f"{path}包含无效BF16十六进制字")
    return np.asarray([int(line, 16) for line in lines], dtype=np.uint16).reshape(shape)


def compare_word_arrays(
    current: np.ndarray,
    candidate: np.ndarray,
    table: str,
) -> list[dict]:
    current = np.asarray(current)
    candidate = np.asarray(candidate)
    if current.dtype != np.uint16 or candidate.dtype != np.uint16:
        raise ValueError("ROM对比输入必须是uint16")
    if current.ndim != 2 or current.shape != candidate.shape:
        raise ValueError("ROM对比输入必须是形状相同的二维数组")
    width = current.shape[1]
    mismatches = []
    for flat_index in np.flatnonzero(current.reshape(-1) != candidate.reshape(-1)):
        index = int(flat_index)
        position, frequency_index = divmod(index, width)
        mismatches.append({
            "table": table,
            "flat_index": index,
            "position": position,
            "frequency_index": frequency_index,
            "current_word": f"{int(current.reshape(-1)[index]):04X}",
            "candidate_word": f"{int(candidate.reshape(-1)[index]):04X}",
        })
    return mismatches


def _numpy_fp64_reference(
    theta: float,
    seq_len: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(seq_len, dtype=np.float64)
    dimensions = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / np.power(theta, dimensions / head_dim)
    angles = np.outer(positions, inv_freq)
    return _bf16_words(np.sin(angles)), _bf16_words(np.cos(angles))


def compare_roms(
    candidate_dir: Path,
    current_rom_dir: Path,
    report_path: Path,
) -> dict:
    """Compare candidate and active ROMs without modifying either directory."""
    candidate_dir = Path(candidate_dir).resolve()
    current_rom_dir = Path(current_rom_dir).resolve()
    report_path = Path(report_path).resolve()
    manifest_path = candidate_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seq_len = int(manifest["seq_len"])
    head_dim = int(manifest["head_dim"])
    shape = (seq_len, head_dim // 2)
    candidate_sin = _read_hex_words(candidate_dir / SIN_FILE, shape)
    candidate_cos = _read_hex_words(candidate_dir / COS_FILE, shape)
    current_sin_path = current_rom_dir / SIN_FILE
    current_cos_path = current_rom_dir / COS_FILE
    current_sin = _read_hex_words(current_sin_path, shape)
    current_cos = _read_hex_words(current_cos_path, shape)
    for key, path in (("sin", candidate_dir / SIN_FILE),
                      ("cos", candidate_dir / COS_FILE)):
        if _sha256_file(path) != manifest["files"][key]["sha256"]:
            raise ValueError(f"候选{key}文件与Manifest哈希不一致")

    mismatches = [
        *compare_word_arrays(current_sin, candidate_sin, "sine"),
        *compare_word_arrays(current_cos, candidate_cos, "cosine"),
    ]
    fp64_sin, fp64_cos = _numpy_fp64_reference(
        float(manifest["rope_theta"]), seq_len, head_dim)
    for item in mismatches:
        reference = fp64_sin if item["table"] == "sine" else fp64_cos
        item["current_matches_numpy_fp64"] = (
            item["current_word"] ==
            f"{int(reference[item['position'], item['frequency_index']]):04X}")

    sin_count = sum(item["table"] == "sine" for item in mismatches)
    cos_count = len(mismatches) - sin_count
    if not mismatches:
        precision_diagnosis = "not_applicable_no_mismatches"
    elif all(item["current_matches_numpy_fp64"] for item in mismatches):
        precision_diagnosis = "current_rom_matches_numpy_fp64"
    else:
        precision_diagnosis = "not_explained_by_numpy_fp64"
    report = {
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "current_rom": {
            "sin_sha256": _sha256_file(current_sin_path),
            "cos_sha256": _sha256_file(current_cos_path),
        },
        "counts": {"sin": sin_count, "cos": cos_count,
                   "total": len(mismatches)},
        "mismatches": mismatches,
        "precision_diagnosis": precision_diagnosis,
        "decision": "match" if not mismatches else "user_confirmation_required",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--compare-rom-dir", type=Path)
    parser.add_argument("--comparison-report", type=Path)
    args = parser.parse_args(argv)
    if (args.compare_rom_dir is None) != (args.comparison_report is None):
        parser.error("--compare-rom-dir and --comparison-report must be used together")
    try:
        manifest = write_lut_bundle(
            args.model_dir / "config.json", args.output_dir,
            Path(__file__).resolve().parents[1] / "mem")
        result = {"manifest": str(manifest)}
        if args.compare_rom_dir is not None:
            report = compare_roms(
                args.output_dir, args.compare_rom_dir, args.comparison_report)
            result.update({"comparison_report": str(args.comparison_report.resolve()),
                           "counts": report["counts"],
                           "decision": report["decision"]})
    except (BundleError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"RoPE LUT generation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
