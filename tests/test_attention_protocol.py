"""TCP v1固定字节布局、CRC和错误语义测试。"""

from dataclasses import replace
from pathlib import Path
import re
import unittest

from python.attention_protocol import (
    CONTEXT_BYTES,
    K_BYTES,
    Q_BYTES,
    REQUEST_HEADER_BYTES,
    REQUEST_STRUCT,
    RESPONSE_HEADER_BYTES,
    RESPONSE_STRUCT,
    V_BYTES,
    ProtocolError,
    RequestHeader,
    ResponseHeader,
    StatusCode,
    crc32,
    pack_request_header,
    pack_response_header,
    unpack_request_header,
    unpack_response_header,
    validate_request_payloads,
)


ROOT = Path(__file__).resolve().parents[1]

REQUEST_VECTOR_HEX = (
    "46505441010128007856341203000000"
    "00001000000004000000040044332211"
    "ddccbbaa04030201"
)
RESPONSE_VECTOR_HEX = (
    "46505441010220007856341200000300"
    "00001000efbeadde6d00000000000000"
)


def request_vector() -> RequestHeader:
    return RequestHeader(
        request_id=0x12345678,
        valid_tokens=3,
        layer_index=0,
        q_bytes=Q_BYTES,
        k_bytes=K_BYTES,
        v_bytes=V_BYTES,
        q_crc32=0x11223344,
        k_crc32=0xAABBCCDD,
        v_crc32=0x01020304,
    )


class AttentionProtocolTest(unittest.TestCase):
    def test_fixed_sizes_and_crc_vector(self):
        self.assertEqual(REQUEST_STRUCT.size, REQUEST_HEADER_BYTES)
        self.assertEqual(RESPONSE_STRUCT.size, RESPONSE_HEADER_BYTES)
        self.assertEqual(REQUEST_HEADER_BYTES, 40)
        self.assertEqual(RESPONSE_HEADER_BYTES, 32)
        self.assertEqual(Q_BYTES, 1_048_576)
        self.assertEqual(K_BYTES, 262_144)
        self.assertEqual(V_BYTES, 262_144)
        self.assertEqual(CONTEXT_BYTES, 1_048_576)
        self.assertEqual(crc32(b"123456789"), 0xCBF43926)

    def test_request_fixed_vector_and_round_trip(self):
        encoded = pack_request_header(request_vector())
        self.assertEqual(encoded.hex(), REQUEST_VECTOR_HEX)
        self.assertEqual(unpack_request_header(encoded), request_vector())

    def test_request_rejects_each_invalid_fixed_field(self):
        original = bytearray(pack_request_header(request_vector()))
        cases = [
            (0, b"BAD!", StatusCode.BAD_MAGIC),
            (4, b"\x02", StatusCode.BAD_VERSION),
            (5, b"\x7f", StatusCode.BAD_HEADER),
            (6, b"\x27\x00", StatusCode.BAD_HEADER),
            (12, b"\x00\x00", StatusCode.BAD_VALID_TOKENS),
            (14, b"\x01\x00", StatusCode.BAD_LAYER_INDEX),
            (16, b"\xff\xff\x0f\x00", StatusCode.BAD_LENGTH),
        ]
        for offset, replacement, status in cases:
            with self.subTest(offset=offset, status=status):
                changed = original.copy()
                changed[offset:offset + len(replacement)] = replacement
                with self.assertRaises(ProtocolError) as raised:
                    unpack_request_header(bytes(changed))
                self.assertEqual(raised.exception.status_code, status)

        for bad_size in (39, 41):
            with self.subTest(size=bad_size):
                with self.assertRaises(ProtocolError) as raised:
                    unpack_request_header(bytes(original[:bad_size]) if bad_size < 40
                                          else bytes(original) + b"\0")
                self.assertEqual(raised.exception.status_code, StatusCode.BAD_HEADER)

    def test_payload_validation_distinguishes_length_and_crc(self):
        q = bytes(Q_BYTES)
        k = bytes(K_BYTES)
        v = bytes(V_BYTES)
        header = replace(
            request_vector(),
            q_crc32=crc32(q),
            k_crc32=crc32(k),
            v_crc32=crc32(v),
        )
        validate_request_payloads(header, q, k, v)

        with self.assertRaises(ProtocolError) as length_error:
            validate_request_payloads(header, q[:-1], k, v)
        self.assertEqual(length_error.exception.status_code, StatusCode.BAD_LENGTH)

        damaged = bytearray(v)
        damaged[-1] = 1
        with self.assertRaises(ProtocolError) as crc_error:
            validate_request_payloads(header, q, k, damaged)
        self.assertEqual(crc_error.exception.status_code, StatusCode.BAD_CRC)

    def test_success_response_fixed_vector_and_round_trip(self):
        header = ResponseHeader(
            request_id=0x12345678,
            status_code=StatusCode.OK,
            valid_tokens=3,
            context_bytes=CONTEXT_BYTES,
            context_crc32=0xDEADBEEF,
            fpga_status=0x6D,
            detail_code=0,
        )
        encoded = pack_response_header(header)
        self.assertEqual(encoded.hex(), RESPONSE_VECTOR_HEX)
        self.assertEqual(unpack_response_header(encoded), header)
        self.assertTrue(unpack_response_header(encoded).ok)

    def test_error_response_has_no_context(self):
        header = ResponseHeader(
            request_id=7,
            status_code=StatusCode.BAD_CRC,
            valid_tokens=3,
            context_bytes=0,
            context_crc32=0,
            fpga_status=0,
            detail_code=2,
        )
        self.assertEqual(unpack_response_header(pack_response_header(header)), header)

        with self.assertRaises(ProtocolError) as raised:
            pack_response_header(replace(header, context_bytes=CONTEXT_BYTES))
        self.assertEqual(raised.exception.status_code, StatusCode.BAD_LENGTH)

    def test_response_rejects_invalid_fields(self):
        good = ResponseHeader(
            request_id=1,
            status_code=StatusCode.OK,
            valid_tokens=1,
            context_bytes=CONTEXT_BYTES,
            context_crc32=1,
            fpga_status=0,
            detail_code=0,
        )
        with self.assertRaises(ProtocolError) as raised:
            pack_response_header(replace(good, valid_tokens=0))
        self.assertEqual(raised.exception.status_code, StatusCode.BAD_VALID_TOKENS)

        raw = bytearray(pack_response_header(good))
        raw[12:14] = b"\xff\xff"
        with self.assertRaises(ProtocolError) as raised:
            unpack_response_header(bytes(raw))
        self.assertEqual(raised.exception.status_code, StatusCode.BAD_HEADER)

    def test_c_header_and_document_share_frozen_constants(self):
        header_text = (ROOT / "vitis" / "attention_server" / "src" /
                       "attention_protocol.h").read_text(encoding="utf-8")

        expected_macros = {
            "FPT_PROTOCOL_VERSION": 1,
            "FPT_REQUEST_HEADER_BYTES": 40,
            "FPT_RESPONSE_HEADER_BYTES": 32,
            "FPT_Q_BYTES": Q_BYTES,
            "FPT_K_BYTES": K_BYTES,
            "FPT_V_BYTES": V_BYTES,
            "FPT_CONTEXT_BYTES": CONTEXT_BYTES,
            "FPT_MAX_VALID_TOKENS": 128,
        }
        for name, expected in expected_macros.items():
            match = re.search(rf"^#define\s+{name}\s+([0-9]+)U$", header_text, re.MULTILINE)
            self.assertIsNotNone(match, name)
            self.assertEqual(int(match.group(1)), expected, name)

        for status in StatusCode:
            self.assertRegex(
                header_text,
                rf"FPT_STATUS_{status.name}\s*=\s*{int(status)}(?:U)?[,\s]",
            )

        document = (ROOT / "docs" / "protocol" /
                    "attention_tcp_v1.md").read_text(encoding="utf-8")
        self.assertIn(REQUEST_VECTOR_HEX, document.replace(" ", "").replace("\n", ""))
        self.assertIn(RESPONSE_VECTOR_HEX, document.replace(" ", "").replace("\n", ""))


if __name__ == "__main__":
    unittest.main()

