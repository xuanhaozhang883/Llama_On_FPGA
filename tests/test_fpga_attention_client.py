"""PC端FPGA Attention客户端的输入、传输和生命周期测试。"""

import hashlib
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from python.attention_protocol import K_BYTES, Q_BYTES, V_BYTES
from python.fpga_attention_client import InputValidationError, load_prepared_inputs


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
