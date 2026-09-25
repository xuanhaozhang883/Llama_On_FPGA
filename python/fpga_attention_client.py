"""Llama第0层FPGA Attention的PC端客户端。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import socket

import numpy as np

from .attention_protocol import (
    K_BYTES,
    MAX_VALID_TOKENS,
    Q_BYTES,
    V_BYTES,
    crc32,
)


class AttentionClientError(RuntimeError):
    """Attention客户端的公共异常基类。"""


class InputValidationError(AttentionClientError):
    """前半层产物不满足固定输入契约。"""


class AttentionTransportError(AttentionClientError):
    """TCP连接、收发或响应帧错误。"""


class AttentionServerError(AttentionClientError):
    """PS端返回非成功状态。"""


@dataclass(frozen=True)
class PreparedInputs:
    valid_tokens: int
    q_payload: bytes
    k_payload: bytes
    v_payload: bytes
    source_dir: Path


_INPUT_FILES = {
    "q": ("q_before_rope_bf16.bin", (32, 128, 128), Q_BYTES),
    "k": ("k_before_rope_bf16.bin", (8, 128, 128), K_BYTES),
    "v": ("v_bf16.bin", (8, 128, 128), V_BYTES),
}
_DTYPE = "BF16 round-to-nearest-even, little-endian uint16"
_LAYOUT = "[head][token][dim] contiguous"


def _validate_prepared_inputs(prepared: PreparedInputs) -> None:
    if not isinstance(prepared, PreparedInputs):
        raise InputValidationError("prepared_inputs类型错误")
    if (type(prepared.valid_tokens) is not int or
            not 1 <= prepared.valid_tokens <= 128):
        raise InputValidationError("valid_tokens必须在1～128之间")
    for name, payload in (
        ("q", prepared.q_payload),
        ("k", prepared.k_payload),
        ("v", prepared.v_payload),
    ):
        _, shape, expected_bytes = _INPUT_FILES[name]
        if not isinstance(payload, bytes) or len(payload) != expected_bytes:
            raise InputValidationError(f"{name}载荷类型或长度错误")
        words = np.frombuffer(payload, dtype="<u2").reshape(shape)
        if np.any(words[:, prepared.valid_tokens:, :] != 0):
            raise InputValidationError(
                f"{name}在valid_tokens之后的补零区不是全零")


def _read_manifest(input_dir: Path) -> dict:
    path = input_dir / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InputValidationError(f"无法读取有效Manifest: {path}") from error
    if not isinstance(value, dict):
        raise InputValidationError("Manifest必须是JSON对象")
    return value


def load_prepared_inputs(input_dir: Path) -> PreparedInputs:
    input_dir = Path(input_dir).resolve()
    manifest = _read_manifest(input_dir)
    if manifest.get("manifest_version") != 2:
        raise InputValidationError("只支持Manifest v2")
    valid_tokens = manifest.get("valid_tokens")
    if (type(valid_tokens) is not int or
            not 1 <= valid_tokens <= MAX_VALID_TOKENS):
        raise InputValidationError("valid_tokens必须在1～128之间")
    if manifest.get("board_seq_len") != 128:
        raise InputValidationError("board_seq_len必须为128")
    token_ids = manifest.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != valid_tokens:
        raise InputValidationError("token_ids数量必须等于valid_tokens")
    if manifest.get("dtype") != _DTYPE or manifest.get("layout") != _LAYOUT:
        raise InputValidationError("Q/K/V数据类型或布局不符合固定接口")

    records = manifest.get("files")
    if not isinstance(records, dict):
        raise InputValidationError("Manifest缺少files对象")
    payloads = {}
    for name, (filename, shape, expected_bytes) in _INPUT_FILES.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise InputValidationError(f"Manifest缺少{name}记录")
        if record.get("file") != filename or record.get("shape") != list(shape):
            raise InputValidationError(f"{name}文件名或形状错误")
        try:
            payload = (input_dir / filename).read_bytes()
        except OSError as error:
            raise InputValidationError(
                f"无法读取{name}载荷: {filename}") from error
        if len(payload) != expected_bytes or record.get("bytes") != expected_bytes:
            raise InputValidationError(f"{name}长度错误")
        expected_crc = record.get("crc32")
        expected_sha = record.get("sha256")
        if (not isinstance(expected_crc, str) or
                not re.fullmatch(r"[0-9a-fA-F]{8}", expected_crc)):
            raise InputValidationError(f"{name} CRC32格式错误")
        if (not isinstance(expected_sha, str) or
                not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha)):
            raise InputValidationError(f"{name} SHA-256格式错误")
        if crc32(payload) != int(expected_crc, 16):
            raise InputValidationError(f"{name} CRC32不匹配")
        if hashlib.sha256(payload).hexdigest() != expected_sha.lower():
            raise InputValidationError(f"{name} SHA-256不匹配")
        payloads[name] = payload

    prepared = PreparedInputs(
        valid_tokens=valid_tokens,
        q_payload=payloads["q"],
        k_payload=payloads["k"],
        v_payload=payloads["v"],
        source_dir=input_dir,
    )
    _validate_prepared_inputs(prepared)
    return prepared
