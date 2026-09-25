"""PC端FPGA Attention客户端的输入、传输和生命周期测试。"""

from dataclasses import replace
import hashlib
import json
import socket
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from python.attention_protocol import (
    CONTEXT_BYTES,
    K_BYTES,
    Q_BYTES,
    RESPONSE_HEADER_BYTES,
    V_BYTES,
    ProtocolError,
    ResponseHeader,
    StatusCode,
    crc32,
    pack_response_header,
    unpack_request_header,
    validate_request_payloads,
)
from python.fpga_attention_client import (
    AttentionServerError,
    AttentionTransportError,
    ContextResponse,
    InputValidationError,
    load_prepared_inputs,
    recv_context_response,
    recv_exact,
    send_qkv_request,
)


def payload_for(heads: int, valid_tokens: int) -> bytes:
    words = np.zeros((heads, 128, 128), dtype="<u2")
    pattern = np.arange(heads * valid_tokens * 128, dtype=np.uint32)
    pattern = (pattern % 0x7FFE + 1).astype("<u2")
    words[:, :valid_tokens, :] = pattern.reshape(heads, valid_tokens, 128)
    return words.tobytes()


def write_prepared_fixture(
        root: Path, valid_tokens: int = 3) -> tuple[Path, dict[str, bytes]]:
    output = root / "front"
    output.mkdir()
    payloads = {
        "q": payload_for(32, valid_tokens),
        "k": payload_for(8, valid_tokens),
        "v": payload_for(8, valid_tokens),
    }
    filenames = {
        "q": "q_before_rope_bf16.bin",
        "k": "k_before_rope_bf16.bin",
        "v": "v_bf16.bin",
    }
    shapes = {
        "q": [32, 128, 128],
        "k": [8, 128, 128],
        "v": [8, 128, 128],
    }
    files = {}
    for name, payload in payloads.items():
        (output / filenames[name]).write_bytes(payload)
        files[name] = {
            "file": filenames[name],
            "bytes": len(payload),
            "crc32": f"{zlib.crc32(payload) & 0xffffffff:08x}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "shape": shapes[name],
        }
    manifest = {
        "manifest_version": 2,
        "token_ids": list(range(valid_tokens)),
        "valid_tokens": valid_tokens,
        "board_seq_len": 128,
        "dtype": "BF16 round-to-nearest-even, little-endian uint16",
        "layout": "[head][token][dim] contiguous",
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    return output, payloads


def rewrite_record(
        output: Path, manifest: dict, name: str, payload: bytes) -> None:
    filename = manifest["files"][name]["file"]
    (output / filename).write_bytes(payload)
    manifest["files"][name]["bytes"] = len(payload)
    manifest["files"][name]["crc32"] = (
        f"{zlib.crc32(payload) & 0xffffffff:08x}")
    manifest["files"][name]["sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")


class ChunkRecvSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.recv_calls = 0
        self.close_calls = 0

    def recv(self, size):
        self.recv_calls += 1
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) <= size:
            return chunk
        self.chunks.insert(0, chunk[size:])
        return chunk[:size]

    def close(self):
        self.close_calls += 1


class RecordingSendSocket:
    def __init__(self, fail_on_call=None, failure=None):
        self.calls = []
        self.fail_on_call = fail_on_call
        self.failure = failure or OSError("simulated send failure")
        self.close_calls = 0

    def sendall(self, data):
        call_number = len(self.calls) + 1
        if call_number == self.fail_on_call:
            raise self.failure
        self.calls.append(bytes(data))

    def close(self):
        self.close_calls += 1


def response_bytes(
        context: bytes,
        request_id=7,
        valid_tokens=3,
        status=StatusCode.OK,
        detail_code=0):
    payload = context if status is StatusCode.OK else b""
    header = ResponseHeader(
        request_id=request_id,
        status_code=status,
        valid_tokens=valid_tokens,
        context_bytes=len(payload),
        context_crc32=crc32(payload) if payload else 0,
        fpga_status=0x6D if status is StatusCode.OK else 0,
        detail_code=detail_code,
    )
    return pack_response_header(header), payload


def overwrite_header_field(raw: bytes, offset: int, replacement: bytes) -> bytes:
    changed = bytearray(raw)
    changed[offset:offset + len(replacement)] = replacement
    return bytes(changed)


class SocketPrimitiveTest(unittest.TestCase):
    def test_recv_exact_accepts_one_byte_and_coalesced_chunks(self):
        self.assertEqual(
            recv_exact(ChunkRecvSocket([b"a", b"b", b"c"]), 3), b"abc")
        self.assertEqual(recv_exact(ChunkRecvSocket([b"abcdef"]), 3), b"abc")

    def test_recv_exact_reports_expected_and_actual_bytes_on_eof(self):
        with self.assertRaisesRegex(
                AttentionTransportError,
                "expected_bytes=5.*actual_bytes=3"):
            recv_exact(ChunkRecvSocket([b"abc", b""]), 5)

    def test_send_qkv_request_writes_exact_header_q_k_v_order(self):
        with tempfile.TemporaryDirectory() as temp:
            output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
            prepared = load_prepared_inputs(output)
            sock = RecordingSendSocket()
            returned = send_qkv_request(sock, prepared, 0x12345678)
            self.assertEqual(len(sock.calls), 4)
            self.assertEqual(len(sock.calls[0]), 40)
            self.assertEqual(
                sock.calls[1:],
                [prepared.q_payload, prepared.k_payload, prepared.v_payload],
            )
            decoded = unpack_request_header(sock.calls[0])
            self.assertEqual(returned, decoded)
            self.assertEqual(decoded.request_id, 0x12345678)
            self.assertEqual(decoded.valid_tokens, prepared.valid_tokens)
            validate_request_payloads(decoded, *sock.calls[1:])

    def test_each_send_failure_preserves_cause_and_diagnostics(self):
        labels = ("请求头", "Q", "K", "V")
        expected_bytes = (40, Q_BYTES, K_BYTES, V_BYTES)
        with tempfile.TemporaryDirectory() as temp:
            output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
            prepared = load_prepared_inputs(output)
            for call_number, (label, size) in enumerate(
                    zip(labels, expected_bytes), start=1):
                with self.subTest(label=label):
                    cause = socket.timeout(f"simulated {label} timeout")
                    sock = RecordingSendSocket(call_number, cause)
                    with self.assertRaises(AttentionTransportError) as raised:
                        send_qkv_request(sock, prepared, 0x12345678)
                    self.assertIs(raised.exception.__cause__, cause)
                    message = str(raised.exception)
                    self.assertIn(label, message)
                    self.assertIn("request_id=305419896", message)
                    self.assertIn(f"expected_bytes={size}", message)
                    self.assertIn("actual_bytes=unknown", message)
                    self.assertEqual(sock.close_calls, 0)

    def test_send_revalidates_manual_prepared_inputs_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
            prepared = load_prepared_inputs(output)
            invalid_length = replace(
                prepared, q_payload=prepared.q_payload[:-2])
            q = bytearray(prepared.q_payload)
            first_padding_word = prepared.valid_tokens * 128 * 2
            q[first_padding_word:first_padding_word + 2] = b"\x01\x00"
            invalid_padding = replace(prepared, q_payload=bytes(q))
            for label, invalid in (
                ("length", invalid_length),
                ("padding", invalid_padding),
            ):
                with self.subTest(label=label):
                    sock = RecordingSendSocket()
                    with self.assertRaises(InputValidationError):
                        send_qkv_request(sock, invalid, 7)
                    self.assertEqual(sock.calls, [])
                    self.assertEqual(sock.close_calls, 0)


class ResponseReceiveTest(unittest.TestCase):
    def test_receives_fragmented_success_response(self):
        context = bytes(range(256)) * (CONTEXT_BYTES // 256)
        header, payload = response_bytes(context)
        chunks = ([bytes([value]) for value in header] +
                  [payload[:123], payload[123:]])
        response = recv_context_response(ChunkRecvSocket(chunks), 7, 3)
        self.assertIsInstance(response, ContextResponse)
        self.assertEqual(response.context_payload, context)

    def test_rejects_each_malformed_response_header_field(self):
        context = bytes(range(256)) * (CONTEXT_BYTES // 256)
        good, _ = response_bytes(context)
        cases = (
            ("magic", 0, b"NOPE"),
            ("version", 4, b"\x02"),
            ("command", 5, b"\x03"),
            ("header_bytes", 6, (31).to_bytes(2, "little")),
            ("context_bytes", 16, (CONTEXT_BYTES - 2).to_bytes(4, "little")),
        )
        for label, offset, replacement in cases:
            with self.subTest(label=label):
                sock = ChunkRecvSocket([
                    overwrite_header_field(good, offset, replacement),
                ])
                with self.assertRaises(AttentionTransportError) as raised:
                    recv_context_response(sock, 7, 3)
                message = str(raised.exception)
                self.assertIn("响应头", message)
                self.assertIn("request_id=7", message)
                self.assertIn("expected_bytes=32", message)
                self.assertEqual(sock.close_calls, 0)

    def test_rejects_mismatched_success_identity(self):
        context = bytes(range(256)) * (CONTEXT_BYTES // 256)
        cases = (
            ("request_id", response_bytes(context, request_id=8)[0]),
            ("valid_tokens", response_bytes(context, valid_tokens=4)[0]),
        )
        for label, raw_header in cases:
            with self.subTest(label=label):
                with self.assertRaises(AttentionTransportError):
                    recv_context_response(
                        ChunkRecvSocket([raw_header]), 7, 3)

    def test_maps_attributable_server_statuses_to_server_error(self):
        cases = (
            (StatusCode.BAD_CRC, 7, 3),
            (StatusCode.BUSY, 7, 3),
            (StatusCode.RUN_TIMEOUT, 7, 3),
            (StatusCode.BAD_MAGIC, 0, 0),
            (StatusCode.INTERNAL_ERROR, 0, 3),
            (StatusCode.RESET_TIMEOUT, 7, 0),
        )
        for status, request_id, valid_tokens in cases:
            with self.subTest(status=status.name):
                raw_header, _ = response_bytes(
                    b"", request_id=request_id, valid_tokens=valid_tokens,
                    status=status, detail_code=11)
                with self.assertRaises(AttentionServerError) as raised:
                    recv_context_response(
                        ChunkRecvSocket([raw_header]), 7, 3)
                self.assertEqual(raised.exception.status_code, status)
                self.assertEqual(raised.exception.request_id, request_id)
                self.assertEqual(raised.exception.valid_tokens, valid_tokens)
                self.assertEqual(raised.exception.fpga_status, 0)
                self.assertEqual(raised.exception.detail_code, 11)

    def test_rejects_nonzero_mismatched_error_identity(self):
        cases = (
            ("request_id", 8, 3),
            ("valid_tokens", 7, 4),
        )
        for label, request_id, valid_tokens in cases:
            with self.subTest(label=label):
                raw_header, _ = response_bytes(
                    b"", request_id=request_id, valid_tokens=valid_tokens,
                    status=StatusCode.BUSY, detail_code=1)
                with self.assertRaises(AttentionTransportError):
                    recv_context_response(
                        ChunkRecvSocket([raw_header]), 7, 3)

    def test_rejects_corrupt_and_truncated_context(self):
        context = bytes(range(256)) * (CONTEXT_BYTES // 256)
        raw_header, payload = response_bytes(context)
        corrupt = bytearray(payload)
        corrupt[-1] ^= 0xFF
        cases = (
            ("crc", [raw_header, bytes(corrupt)], "CRC32"),
            ("truncated", [raw_header, payload[:12345], b""],
             "actual_bytes=12345"),
        )
        for label, chunks, diagnostic in cases:
            with self.subTest(label=label):
                with self.assertRaises(AttentionTransportError) as raised:
                    recv_context_response(ChunkRecvSocket(chunks), 7, 3)
                self.assertIn("request_id=7", str(raised.exception))
                self.assertIn(diagnostic, str(raised.exception))

    def test_rejects_nonzero_detail_code_in_success_header(self):
        context = bytes(range(256)) * (CONTEXT_BYTES // 256)
        good, _ = response_bytes(context)
        invalid = overwrite_header_field(
            good, 28, (1).to_bytes(4, "little"))
        with self.assertRaises(AttentionTransportError) as raised:
            recv_context_response(ChunkRecvSocket([invalid]), 7, 3)
        self.assertIsInstance(raised.exception.__cause__, ProtocolError)

    def test_rejects_invalid_expected_identity_before_socket_read(self):
        cases = ((-1, 3), (7, 0))
        for expected_request_id, expected_valid_tokens in cases:
            with self.subTest(
                    request_id=expected_request_id,
                    valid_tokens=expected_valid_tokens):
                sock = ChunkRecvSocket([])
                with self.assertRaises(InputValidationError):
                    recv_context_response(
                        sock, expected_request_id, expected_valid_tokens)
                self.assertEqual(sock.recv_calls, 0)

    def test_low_level_failures_do_not_close_callers_socket(self):
        context = bytes(range(256)) * (CONTEXT_BYTES // 256)
        with tempfile.TemporaryDirectory() as temp:
            output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
            prepared = load_prepared_inputs(output)

            send_sock = RecordingSendSocket(
                fail_on_call=2, failure=OSError("simulated Q failure"))
            with self.assertRaises(AttentionTransportError):
                send_qkv_request(send_sock, prepared, 7)
            self.assertEqual(send_sock.close_calls, 0)

            mismatched_header, _ = response_bytes(context, request_id=8)
            identity_sock = ChunkRecvSocket([mismatched_header])
            with self.assertRaises(AttentionTransportError):
                recv_context_response(identity_sock, 7, 3)
            self.assertEqual(identity_sock.close_calls, 0)

            crc_header, payload = response_bytes(context)
            corrupted = bytearray(payload)
            corrupted[-1] ^= 0xFF
            crc_sock = ChunkRecvSocket([crc_header, bytes(corrupted)])
            with self.assertRaises(AttentionTransportError):
                recv_context_response(crc_sock, 7, 3)
            self.assertEqual(crc_sock.close_calls, 0)


class PreparedInputsTest(unittest.TestCase):
    def test_loads_verified_manifest_and_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            output, payloads = write_prepared_fixture(Path(temp), valid_tokens=3)
            self.assertEqual(len(payloads["q"]), Q_BYTES)
            self.assertEqual(len(payloads["k"]), K_BYTES)
            self.assertEqual(len(payloads["v"]), V_BYTES)

            prepared = load_prepared_inputs(output)
            self.assertEqual(prepared.valid_tokens, 3)
            self.assertEqual(prepared.q_payload, payloads["q"])
            self.assertEqual(prepared.k_payload, payloads["k"])
            self.assertEqual(prepared.v_payload, payloads["v"])
            self.assertEqual(prepared.source_dir, output.resolve())

    def test_rejects_manifest_identity_and_file_integrity_errors(self):
        mutations = (
            ("manifest_version", lambda m: m.__setitem__("manifest_version", 1)),
            ("valid_tokens_zero", lambda m: m.__setitem__("valid_tokens", 0)),
            ("valid_tokens_too_large", lambda m: m.__setitem__("valid_tokens", 129)),
            ("token_ids", lambda m: m.__setitem__("token_ids", [])),
            ("board_seq_len", lambda m: m.__setitem__("board_seq_len", 127)),
            ("dtype", lambda m: m.__setitem__("dtype", "FP16")),
            ("layout", lambda m: m.__setitem__("layout", "[token][head][dim]")),
            ("filename", lambda m: m["files"]["q"].__setitem__("file", "wrong.bin")),
            ("shape", lambda m: m["files"]["k"].__setitem__("shape", [7, 128, 128])),
            ("bytes", lambda m: m["files"]["v"].__setitem__("bytes", V_BYTES - 2)),
            ("crc", lambda m: m["files"]["q"].__setitem__("crc32", "00000000")),
            ("sha", lambda m: m["files"]["k"].__setitem__("sha256", "0" * 64)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
                manifest_path = output / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(InputValidationError):
                    load_prepared_inputs(output)

    def test_rejects_invalid_json_and_non_object_manifest(self):
        documents = (
            ("invalid_json", "{"),
            ("json_array", "[]"),
        )
        for label, document in documents:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
                (output / "manifest.json").write_text(document, encoding="utf-8")
                with self.assertRaises(InputValidationError):
                    load_prepared_inputs(output)

    def test_rejects_each_missing_required_file(self):
        required = (
            "manifest.json",
            "q_before_rope_bf16.bin",
            "k_before_rope_bf16.bin",
            "v_bf16.bin",
        )
        for filename in required:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
                (output / filename).unlink()
                with self.assertRaises(InputValidationError):
                    load_prepared_inputs(output)

    def test_rejects_truncated_payload_even_when_manifest_digests_match(self):
        with tempfile.TemporaryDirectory() as temp:
            output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            truncated = (output / "v_bf16.bin").read_bytes()[:-2]
            rewrite_record(output, manifest, "v", truncated)
            with self.assertRaisesRegex(InputValidationError, "长度"):
                load_prepared_inputs(output)

    def test_rejects_nonzero_words_after_valid_tokens_even_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8"))
            q = bytearray((output / "q_before_rope_bf16.bin").read_bytes())
            first_invalid_word = ((0 * 128 + 3) * 128) * 2
            q[first_invalid_word:first_invalid_word + 2] = b"\x01\x00"
            rewrite_record(output, manifest, "q", bytes(q))
            with self.assertRaisesRegex(InputValidationError, "补零"):
                load_prepared_inputs(output)


if __name__ == "__main__":
    unittest.main()
