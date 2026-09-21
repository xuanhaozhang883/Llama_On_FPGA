"""Synthetic weights exercise the real safetensors loading and DDR exporter."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from python.prepare_llama3_qkv import bf16_le_bytes, project_layer0, write_outputs


class PrepareLlama3QkvTest(unittest.TestCase):
    def test_projection_layout_padding_and_bf16(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model"
            output = Path(td) / "output"
            model.mkdir()
            cfg = {"model_type": "llama", "hidden_size": 8,
                   "num_attention_heads": 2, "num_key_value_heads": 1,
                   "head_dim": 4, "rms_norm_eps": 1e-5}
            (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            embedding = torch.arange(10 * 8, dtype=torch.float32).reshape(10, 8) / 16
            norm = torch.arange(1, 9, dtype=torch.float32) / 8
            q_weight = torch.eye(8)
            k_weight = torch.eye(8)[:4]
            v_weight = torch.flip(torch.eye(8), dims=[0])[:4]
            save_file({
                "model.embed_tokens.weight": embedding.to(torch.bfloat16),
                "model.layers.0.input_layernorm.weight": norm.to(torch.bfloat16),
                "model.layers.0.self_attn.q_proj.weight": q_weight.to(torch.bfloat16),
                "model.layers.0.self_attn.k_proj.weight": k_weight.to(torch.bfloat16),
                "model.layers.0.self_attn.v_proj.weight": v_weight.to(torch.bfloat16),
            }, model / "model.safetensors")
            board = {"q_heads": 2, "kv_heads": 1, "head_dim": 4,
                     "seq_len": 4, "q_base": "0x10000000",
                     "k_base": "0x10100000", "v_base": "0x10140000"}
            ids = [3, 1]
            arrays = project_layer0(model, ids, board)
            x = embedding[ids]
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * norm
            for name, weight, heads in (("q", q_weight, 2),
                                        ("k", k_weight, 1),
                                        ("v", v_weight, 1)):
                expected = (x @ weight.T).reshape(2, heads, 4).permute(1, 0, 2).numpy()
                np.testing.assert_allclose(arrays[name][:, :2, :], expected, rtol=1e-6)
                self.assertTrue(np.all(arrays[name][:, 2:, :] == 0))

            write_outputs(output, arrays, ids, board)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["valid_tokens"], 2)
            for name, heads in (("q", 2), ("k", 1), ("v", 1)):
                item = manifest["files"][name]
                payload = (output / item["file"]).read_bytes()
                self.assertEqual(len(payload), heads * 4 * 4 * 2)
                expected = torch.from_numpy(arrays[name]).to(torch.bfloat16)
                self.assertEqual(payload, expected.view(torch.uint16).numpy().astype("<u2").tobytes())

    def test_tie_rounds_to_even(self):
        samples = np.array([1.0 + 2**-8, 1.0 + 3 * 2**-8], dtype=np.float32)
        words = np.frombuffer(bf16_le_bytes(samples), dtype="<u2")
        np.testing.assert_array_equal(words, [0x3f80, 0x3f82])


if __name__ == "__main__":
    unittest.main()
