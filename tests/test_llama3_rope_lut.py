"""Llama 3 RoPE LUT generation, layout, safety, and comparison tests."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from python.generate_llama3_rope_lut import (
    compare_roms,
    compare_word_arrays,
    generate_rope_words,
    write_lut_bundle,
)
from python.prepare_llama3_model import BundleError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16",
    "rope_theta": 500000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "mlp_bias": False,
}


def torch_reference(config, seq_len=128, head_dim=128):
    positions = torch.arange(seq_len, dtype=torch.float32)
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (float(config["rope_theta"]) ** (dimensions / head_dim))
    angles = torch.outer(positions, inv_freq)
    sine = angles.sin().to(torch.bfloat16).view(torch.uint16).numpy()
    cosine = angles.cos().to(torch.bfloat16).view(torch.uint16).numpy()
    return sine, cosine


class Llama3RopeLutTest(unittest.TestCase):
    def test_full_lut_matches_independent_torch_reference(self):
        actual_sin, actual_cos = generate_rope_words(CONFIG)
        expected_sin, expected_cos = torch_reference(CONFIG)
        np.testing.assert_array_equal(actual_sin, expected_sin)
        np.testing.assert_array_equal(actual_cos, expected_cos)
        self.assertEqual(actual_sin.shape, (128, 64))
        self.assertEqual(actual_cos.shape, (128, 64))

    def test_position_zero_and_flattening_order(self):
        sine, cosine = generate_rope_words(CONFIG)
        np.testing.assert_array_equal(sine[0], np.zeros(64, dtype=np.uint16))
        np.testing.assert_array_equal(
            cosine[0], np.full(64, 0x3F80, dtype=np.uint16))

    def test_writer_refuses_repository_mem_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mem"):
                write_lut_bundle(config_path, ROOT / "mem", ROOT / "mem")

    def test_scaled_rope_is_rejected(self):
        scaled = {**CONFIG, "rope_scaling": {"rope_type": "llama3"}}
        with self.assertRaises(BundleError):
            generate_rope_words(scaled)

    def test_compare_reports_word_index_position_and_frequency(self):
        report = compare_word_arrays(
            np.array([[0x0000, 0x3F80]], dtype=np.uint16),
            np.array([[0x0000, 0x3F81]], dtype=np.uint16),
            "cosine",
        )
        self.assertEqual(report[0], {
            "table": "cosine",
            "flat_index": 1,
            "position": 0,
            "frequency_index": 1,
            "current_word": "3F80",
            "candidate_word": "3F81",
        })

    def test_writer_and_rom_comparison_create_auditable_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
            candidate = root / "candidate"
            manifest_path = write_lut_bundle(
                config_path, candidate, forbidden_rom_dir=ROOT / "mem")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["algorithm"], "torch_fp32_then_bf16_rne")
            self.assertEqual(manifest["element_count_per_table"], 8192)
            self.assertEqual(len((candidate / "sin_bf16.hex").read_text().splitlines()), 8192)
            self.assertEqual(len((candidate / "cos_bf16.hex").read_text().splitlines()), 8192)
            self.assertEqual(list(candidate.glob("*.tmp")), [])

            current = root / "current"
            current.mkdir()
            shutil.copy2(candidate / "sin_bf16.hex", current / "sin_bf16.hex")
            cosine_lines = (candidate / "cos_bf16.hex").read_text().splitlines()
            cosine_lines[1] = "3F81"
            (current / "cos_bf16.hex").write_text(
                "\n".join(cosine_lines) + "\n", encoding="ascii")
            report_path = root / "report" / "rom-comparison.json"
            report = compare_roms(candidate, current, report_path)
            self.assertEqual(report["counts"], {"sin": 0, "cos": 1, "total": 1})
            self.assertEqual(report["decision"], "user_confirmation_required")
            self.assertFalse(report["mismatches"][0]["current_matches_numpy_fp64"])
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
