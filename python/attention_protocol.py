"""Llama 第 0 层 PC/PS Attention TCP v1 协议。

本模块只负责固定长度头部、CRC 和字段校验，不负责 socket 收发。
所有多字节整数均使用小端编码，载荷顺序固定为 Q、K、V。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
import zlib


MAGIC = b"FPTA"
VERSION = 1
RUN_ATTENTION = 1
ATTENTION_RESULT = 2

REQUEST_HEADER_BYTES = 40
RESPONSE_HEADER_BYTES = 32

Q_BYTES = 32 * 128 * 128 * 2
K_BYTES = 8 * 128 * 128 * 2
V_BYTES = 8 * 128 * 128 * 2
REQUEST_PAYLOAD_BYTES = Q_BYTES + K_BYTES + V_BYTES
CONTEXT_BYTES = 32 * 128 * 128 * 2

MIN_VALID_TOKENS = 1
MAX_VALID_TOKENS = 128
LAYER_INDEX = 0

REQUEST_STRUCT = struct.Struct("<4sBBHIHHIIIIII")
RESPONSE_STRUCT = struct.Struct("<4sBBHIHHIIII")

assert REQUEST_STRUCT.size == REQUEST_HEADER_BYTES
assert RESPONSE_STRUCT.size == RESPONSE_HEADER_BYTES


class StatusCode(IntEnum):
    OK = 0
    BAD_MAGIC = 1
    BAD_VERSION = 2
    BAD_HEADER = 3
    BAD_LENGTH = 4
    BAD_VALID_TOKENS = 5
    BAD_LAYER_INDEX = 6
    BAD_CRC = 7
    BUSY = 8
    RESET_TIMEOUT = 9
    RUN_TIMEOUT = 10
    FPGA_ERROR = 11
    INTERNAL_ERROR = 12


class ProtocolError(ValueError):
    """表示可以映射到TCP v1状态码的协议错误。"""

    def __init__(self, status_code: StatusCode, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class RequestHeader:
    request_id: int
    valid_tokens: int
    layer_index: int
    q_bytes: int
    k_bytes: int
    v_bytes: int
    q_crc32: int
    k_crc32: int
    v_crc32: int


@dataclass(frozen=True)
class ResponseHeader:
    request_id: int
    status_code: StatusCode
    valid_tokens: int
    context_bytes: int
    context_crc32: int
    fpga_status: int
    detail_code: int

    @property
    def ok(self) -> bool:
        return self.status_code is StatusCode.OK


def crc32(data: bytes | bytearray | memoryview) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _require_uint(name: str, value: int, bits: int) -> None:
    if type(value) is not int or not 0 <= value < (1 << bits):
        raise ValueError(f"{name}必须是u{bits}范围内的整数")


def _validate_request_fields(header: RequestHeader) -> None:
    _require_uint("request_id", header.request_id, 32)
    if not MIN_VALID_TOKENS <= header.valid_tokens <= MAX_VALID_TOKENS:
        raise ProtocolError(StatusCode.BAD_VALID_TOKENS, "valid_tokens必须在1～128之间")
    if header.layer_index != LAYER_INDEX:
        raise ProtocolError(StatusCode.BAD_LAYER_INDEX, "TCP v1只支持layer_index=0")
    if (header.q_bytes, header.k_bytes, header.v_bytes) != (Q_BYTES, K_BYTES, V_BYTES):
        raise ProtocolError(StatusCode.BAD_LENGTH, "Q/K/V长度不是TCP v1固定长度")
    for name, value in (("q_crc32", header.q_crc32),
                        ("k_crc32", header.k_crc32),
                        ("v_crc32", header.v_crc32)):
        _require_uint(name, value, 32)


def pack_request_header(header: RequestHeader) -> bytes:
    _validate_request_fields(header)
    return REQUEST_STRUCT.pack(
        MAGIC,
        VERSION,
        RUN_ATTENTION,
        REQUEST_HEADER_BYTES,
        header.request_id,
        header.valid_tokens,
        header.layer_index,
        header.q_bytes,
        header.k_bytes,
        header.v_bytes,
        header.q_crc32,
        header.k_crc32,
        header.v_crc32,
    )


def make_request_header(
    *, request_id: int, valid_tokens: int,
    q_payload: bytes | bytearray | memoryview,
    k_payload: bytes | bytearray | memoryview,
    v_payload: bytes | bytearray | memoryview,
) -> RequestHeader:
    header = RequestHeader(
        request_id=request_id,
        valid_tokens=valid_tokens,
        layer_index=LAYER_INDEX,
        q_bytes=len(q_payload),
        k_bytes=len(k_payload),
        v_bytes=len(v_payload),
        q_crc32=crc32(q_payload),
        k_crc32=crc32(k_payload),
        v_crc32=crc32(v_payload),
    )
    _validate_request_fields(header)
    return header


def unpack_request_header(data: bytes) -> RequestHeader:
    if len(data) != REQUEST_HEADER_BYTES:
        raise ProtocolError(StatusCode.BAD_HEADER, "请求头必须恰好为40字节")

    (magic, version, command, header_bytes, request_id, valid_tokens,
     layer_index, q_bytes, k_bytes, v_bytes, q_crc, k_crc, v_crc) = REQUEST_STRUCT.unpack(data)

    if magic != MAGIC:
        raise ProtocolError(StatusCode.BAD_MAGIC, "请求Magic不是FPTA")
    if version != VERSION:
        raise ProtocolError(StatusCode.BAD_VERSION, "不支持的协议版本")
    if command != RUN_ATTENTION or header_bytes != REQUEST_HEADER_BYTES:
        raise ProtocolError(StatusCode.BAD_HEADER, "请求命令或header_bytes错误")

    header = RequestHeader(
        request_id=request_id,
        valid_tokens=valid_tokens,
        layer_index=layer_index,
        q_bytes=q_bytes,
        k_bytes=k_bytes,
        v_bytes=v_bytes,
        q_crc32=q_crc,
        k_crc32=k_crc,
        v_crc32=v_crc,
    )
    _validate_request_fields(header)
    return header


def validate_request_payloads(
    header: RequestHeader,
    q_payload: bytes | bytearray | memoryview,
    k_payload: bytes | bytearray | memoryview,
    v_payload: bytes | bytearray | memoryview,
) -> None:
    actual_lengths = (len(q_payload), len(k_payload), len(v_payload))
    expected_lengths = (header.q_bytes, header.k_bytes, header.v_bytes)
    if actual_lengths != expected_lengths:
        raise ProtocolError(StatusCode.BAD_LENGTH, "实际Q/K/V载荷长度与请求头不一致")
    actual_crc = (crc32(q_payload), crc32(k_payload), crc32(v_payload))
    expected_crc = (header.q_crc32, header.k_crc32, header.v_crc32)
    if actual_crc != expected_crc:
        raise ProtocolError(StatusCode.BAD_CRC, "Q/K/V载荷CRC错误")


def _validate_response_fields(header: ResponseHeader) -> None:
    _require_uint("request_id", header.request_id, 32)
    _require_uint("fpga_status", header.fpga_status, 32)
    _require_uint("detail_code", header.detail_code, 32)
    _require_uint("context_crc32", header.context_crc32, 32)

    try:
        status = StatusCode(header.status_code)
    except ValueError as exc:
        raise ProtocolError(StatusCode.BAD_HEADER, "未知status_code") from exc

    if status is StatusCode.OK:
        if header.detail_code != 0:
            raise ProtocolError(StatusCode.BAD_HEADER, "成功响应的detail_code必须为0")
        if not MIN_VALID_TOKENS <= header.valid_tokens <= MAX_VALID_TOKENS:
            raise ProtocolError(StatusCode.BAD_VALID_TOKENS, "成功响应的valid_tokens必须在1～128之间")
        if header.context_bytes != CONTEXT_BYTES:
            raise ProtocolError(StatusCode.BAD_LENGTH, "成功响应必须携带完整1 MiB Context")
    else:
        if not 0 <= header.valid_tokens <= MAX_VALID_TOKENS:
            raise ProtocolError(StatusCode.BAD_VALID_TOKENS, "错误响应的valid_tokens必须在0～128之间")
        if header.context_bytes != 0 or header.context_crc32 != 0:
            raise ProtocolError(StatusCode.BAD_LENGTH, "错误响应不能携带Context")


def pack_response_header(header: ResponseHeader) -> bytes:
    _validate_response_fields(header)
    return RESPONSE_STRUCT.pack(
        MAGIC,
        VERSION,
        ATTENTION_RESULT,
        RESPONSE_HEADER_BYTES,
        header.request_id,
        int(header.status_code),
        header.valid_tokens,
        header.context_bytes,
        header.context_crc32,
        header.fpga_status,
        header.detail_code,
    )


def unpack_response_header(data: bytes) -> ResponseHeader:
    if len(data) != RESPONSE_HEADER_BYTES:
        raise ProtocolError(StatusCode.BAD_HEADER, "响应头必须恰好为32字节")

    (magic, version, command, header_bytes, request_id, status_code,
     valid_tokens, context_bytes, context_crc, fpga_status,
     detail_code) = RESPONSE_STRUCT.unpack(data)

    if magic != MAGIC:
        raise ProtocolError(StatusCode.BAD_MAGIC, "响应Magic不是FPTA")
    if version != VERSION:
        raise ProtocolError(StatusCode.BAD_VERSION, "不支持的协议版本")
    if command != ATTENTION_RESULT or header_bytes != RESPONSE_HEADER_BYTES:
        raise ProtocolError(StatusCode.BAD_HEADER, "响应命令或header_bytes错误")

    try:
        status = StatusCode(status_code)
    except ValueError as exc:
        raise ProtocolError(StatusCode.BAD_HEADER, "未知status_code") from exc

    header = ResponseHeader(
        request_id=request_id,
        status_code=status,
        valid_tokens=valid_tokens,
        context_bytes=context_bytes,
        context_crc32=context_crc,
        fpga_status=fpga_status,
        detail_code=detail_code,
    )
    _validate_response_fields(header)
    return header

