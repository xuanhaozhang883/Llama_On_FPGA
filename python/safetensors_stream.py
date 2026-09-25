"""Read selected safetensors bytes without mapping a multi-gigabyte shard.

On Windows, mapping the whole Llama shard can fail with os error 1455 when the
page file is small. The safetensors header gives byte offsets for each tensor,
so ordinary seek/read calls can load only the rows or tensors needed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
_MAX_HEADER_BYTES = 100_000_000


def _decode_fp32(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2").astype("<u4")
        return (words << 16).view("<f4").reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
    raise ValueError(f"不支持的权重类型: {dtype}")


class SafeTensorReader:
    """Small, read-only subset of safetensors for BF16/F16/F32 model weights."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        try:
            prefix = self.file.read(8)
            if len(prefix) != 8:
                raise ValueError(f"权重文件头不完整: {self.path}")
            header_size = int.from_bytes(prefix, "little")
            if not 2 <= header_size <= _MAX_HEADER_BYTES:
                raise ValueError(f"权重文件头长度无效: {self.path}")
            raw_header = self.file.read(header_size)
            if len(raw_header) != header_size or not raw_header.startswith(b"{"):
                raise ValueError(f"权重文件头不完整或格式错误: {self.path}")
            self.header = json.loads(raw_header)
            if not isinstance(self.header, dict):
                raise ValueError(f"权重文件头格式错误: {self.path}")
            self.data_start = 8 + header_size
            self.file_size = self.path.stat().st_size
            tensor_ends = [entry["data_offsets"][1]
                           for key, entry in self.header.items() if key != "__metadata__"]
            if not tensor_ends or self.data_start + max(tensor_ends) != self.file_size:
                raise ValueError(f"权重文件大小与文件头不一致，可能未下载完整: {self.path}")
        except Exception:
            self.file.close()
            raise

    def __enter__(self) -> "SafeTensorReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.file.close()

    def _tensor(self, key: str) -> tuple[str, tuple[int, ...], int, int]:
        if key not in self.header:
            raise ValueError(f"{self.path} 中缺少 {key}")
        entry = self.header[key]
        dtype = entry["dtype"]
        shape = entry["shape"]
        offsets = entry["data_offsets"]
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"{key} 的类型 {dtype} 暂不支持")
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)):
            raise ValueError(f"{key} 的维度或偏移无效")
        begin, end = offsets
        elements = 1
        for n in shape:
            elements *= n
        if (begin < 0 or end < begin or end > self.file_size - self.data_start
                or end - begin != elements * _DTYPE_BYTES[dtype]):
            raise ValueError(f"{key} 的数据长度或偏移无效")
        return dtype, tuple(shape), begin, end

    def shape(self, key: str) -> tuple[int, ...]:
        return self._tensor(key)[1]

    def _read(self, offset: int, size: int) -> bytes:
        self.file.seek(offset)
        raw = self.file.read(size)
        if len(raw) != size:
            raise ValueError(f"读取权重失败，文件可能不完整: {self.path}")
        return raw

    def read_tensor_fp32(self, key: str) -> np.ndarray:
        dtype, shape, begin, end = self._tensor(key)
        return _decode_fp32(self._read(self.data_start + begin, end - begin), dtype, shape)

    def read_row_range_fp32(self, key: str, start: int, end: int) -> np.ndarray:
        dtype, shape, begin, _ = self._tensor(key)
        if len(shape) != 2 or not 0 <= start <= end <= shape[0]:
            raise ValueError(f"{key} 的读取行号超出范围")
        row_bytes = shape[1] * _DTYPE_BYTES[dtype]
        raw = self._read(self.data_start + begin + start * row_bytes,
                         (end - start) * row_bytes)
        return _decode_fp32(raw, dtype, (end - start, shape[1]))

    def read_rows_fp32(self, key: str, ids: list[int]) -> np.ndarray:
        shape = self.shape(key)
        if len(shape) != 2 or any(type(i) is not int or i < 0 or i >= shape[0] for i in ids):
            raise ValueError(f"Token ID 超出权重词表大小 {shape[0]}")
        if not ids:
            return np.empty((0, shape[1]), dtype=np.float32)
        unique = {i: self.read_row_range_fp32(key, i, i + 1)[0]
                  for i in sorted(set(ids))}
        return np.stack([unique[i] for i in ids]).astype(np.float32, copy=False)
