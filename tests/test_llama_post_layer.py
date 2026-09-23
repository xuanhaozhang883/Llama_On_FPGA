"""Context拼头和Llama第0层后半段的独立CPU FP32测试。"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from python.llama_post_layer import (
    PostWeights,
    decode_context_bf16,
    load_post_weights,
    run_post_attention,
)


def bf16_payload(values: np.ndarray) -> bytes:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(torch.bfloat16)
    return tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()


def small_weights(hidden: int = 4, intermediate: int = 6) -> PostWeights:
    o_proj = torch.tensor([
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.0, 0.0, -0.5],
        [0.25, 0.0, 1.0, 0.0],
        [0.0, -0.25, 0.0, 1.0],
    ], dtype=torch.float32)
    gate = torch.arange(1, intermediate * hidden + 1, dtype=torch.float32).reshape(intermediate, hidden) / 20
    up = torch.flip(gate, dims=[0]) / 2
    down = torch.arange(1, hidden * intermediate + 1, dtype=torch.float32).reshape(hidden, intermediate) / 30
    return PostWeights(
        o_proj=o_proj,
        post_attention_layernorm=torch.tensor([1.0, 0.5, 1.5, 2.0]),
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
    )


class LlamaPostLayerTest(unittest.TestCase):
    def test_decode_context_maps_head_token_dim_to_token_hidden(self):
        heads, seq_len, head_dim, valid_tokens = 2, 4, 3, 3
        source = np.empty((heads, seq_len, head_dim), dtype=np.float32)
        for head in range(heads):
            for token in range(seq_len):
                for dim in range(head_dim):
                    source[head, token, dim] = head * 64 + token * 8 + dim

        decoded = decode_context_bf16(
            bf16_payload(source),
            valid_tokens=valid_tokens,
            q_heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )

        self.assertEqual(tuple(decoded.shape), (1, valid_tokens, heads * head_dim))
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertEqual(decoded.device.type, "cpu")
        for token in range(valid_tokens):
            expected = [source[head, token, dim]
                        for head in range(heads) for dim in range(head_dim)]
            np.testing.assert_array_equal(decoded[0, token].numpy(), expected)

    def test_decode_context_rejects_bad_length_tokens_and_nonfinite_values(self):
        good = np.zeros((2, 4, 3), dtype=np.float32)
        payload = bf16_payload(good)
        for bad_payload in (payload[:-2], payload + b"\0\0"):
            with self.subTest(size=len(bad_payload)), self.assertRaisesRegex(ValueError, "48"):
                decode_context_bf16(
                    bad_payload, valid_tokens=1, q_heads=2, seq_len=4, head_dim=3)

        for valid_tokens in (0, 5):
            with self.subTest(valid_tokens=valid_tokens), self.assertRaisesRegex(ValueError, "1～4"):
                decode_context_bf16(
                    payload, valid_tokens=valid_tokens, q_heads=2, seq_len=4, head_dim=3)

        bad = good.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN|无穷"):
            decode_context_bf16(
                bf16_payload(bad), valid_tokens=1, q_heads=2, seq_len=4, head_dim=3)

    def test_run_post_attention_matches_independent_numpy_checkpoints(self):
        weights = small_weights()
        context = torch.tensor([[[0.5, -1.0, 2.0, 0.25],
                                 [1.5, 0.5, -0.5, 2.0]]], dtype=torch.float32)
        residual = torch.tensor([[[2.0, 1.0, 0.5, -1.0],
                                  [-0.5, 1.25, 2.0, 0.75]]], dtype=torch.float32)
        eps = 1e-5

        result = run_post_attention(context, residual, weights, rms_eps=eps)

        x = context.numpy()
        r = residual.numpy()
        o = x @ weights.o_proj.numpy().T
        residual_1 = o + r
        rms = np.sqrt(np.mean(np.square(residual_1), axis=-1, keepdims=True) + eps)
        post_norm = residual_1 / rms * weights.post_attention_layernorm.numpy()
        gate = post_norm @ weights.gate_proj.numpy().T
        up = post_norm @ weights.up_proj.numpy().T
        swiglu = (gate / (1.0 + np.exp(-gate))) * up
        down = swiglu @ weights.down_proj.numpy().T
        expected = {
            "context_merged": x,
            "o_proj": o,
            "residual_1": residual_1,
            "post_norm": post_norm,
            "gate": gate,
            "up": up,
            "swiglu": swiglu,
            "down_proj": down,
            "layer_output": residual_1 + down,
        }
        for name, value in expected.items():
            with self.subTest(checkpoint=name):
                np.testing.assert_allclose(
                    getattr(result, name).numpy(), value, rtol=1e-5, atol=1e-6)

    def test_run_post_attention_supports_one_short_and_128_tokens(self):
        hidden = 4
        identity = torch.eye(hidden, dtype=torch.float32)
        weights = PostWeights(
            o_proj=identity,
            post_attention_layernorm=torch.ones(hidden),
            gate_proj=identity,
            up_proj=identity * 2,
            down_proj=identity,
        )
        for length in (1, 3, 128):
            with self.subTest(length=length):
                context = torch.arange(length * hidden, dtype=torch.float32).reshape(1, length, hidden) / 16
                residual = torch.full_like(context, 0.25)
                result = run_post_attention(context, residual, weights, rms_eps=1e-5)
                self.assertEqual(tuple(result.layer_output.shape), (1, length, hidden))
                self.assertTrue(torch.isfinite(result.layer_output).all())

    def test_run_post_attention_rejects_shape_dtype_and_nonfinite_inputs(self):
        weights = small_weights()
        context = torch.zeros((1, 2, 4), dtype=torch.float32)
        residual = torch.zeros_like(context)

        bad_cases = [
            (context[:, :, :3], residual[:, :, :3], "最后一维"),
            (context, residual[:, :1, :], "形状"),
            (context.to(torch.float64), residual, "FP32"),
        ]
        for bad_context, bad_residual, message in bad_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                run_post_attention(bad_context, bad_residual, weights, rms_eps=1e-5)

        nonfinite = context.clone()
        nonfinite[0, 0, 0] = math.inf
        with self.assertRaisesRegex(ValueError, "NaN|无穷"):
            run_post_attention(nonfinite, residual, weights, rms_eps=1e-5)

        bad_weights = PostWeights(
            o_proj=weights.o_proj,
            post_attention_layernorm=weights.post_attention_layernorm,
            gate_proj=weights.gate_proj,
            up_proj=weights.up_proj,
            down_proj=weights.down_proj[:, :-1],
        )
        with self.assertRaisesRegex(ValueError, "down_proj"):
            run_post_attention(context, residual, bad_weights, rms_eps=1e-5)

    def test_load_post_weights_reads_only_required_layer_tensors(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)
            hidden, intermediate = 4, 6
            config = {
                "model_type": "llama",
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "attention_bias": False,
                "mlp_bias": False,
            }
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            expected = small_weights(hidden, intermediate)
            save_file({
                "model.layers.0.self_attn.o_proj.weight": expected.o_proj.to(torch.bfloat16),
                "model.layers.0.post_attention_layernorm.weight": expected.post_attention_layernorm.to(torch.bfloat16),
                "model.layers.0.mlp.gate_proj.weight": expected.gate_proj.to(torch.bfloat16),
                "model.layers.0.mlp.up_proj.weight": expected.up_proj.to(torch.bfloat16),
                "model.layers.0.mlp.down_proj.weight": expected.down_proj.to(torch.bfloat16),
                "model.layers.1.self_attn.o_proj.weight": torch.zeros((hidden, hidden), dtype=torch.bfloat16),
            }, model / "model.safetensors")

            loaded = load_post_weights(model, layer_idx=0)

            for name in ("o_proj", "post_attention_layernorm", "gate_proj", "up_proj", "down_proj"):
                with self.subTest(weight=name):
                    self.assertEqual(getattr(loaded, name).dtype, torch.float32)
                    np.testing.assert_allclose(
                        getattr(loaded, name).numpy(),
                        getattr(expected, name).to(torch.bfloat16).float().numpy(),
                        rtol=0,
                        atol=0,
                    )

            with self.assertRaisesRegex(ValueError, "layer_idx"):
                load_post_weights(model, layer_idx=2)


if __name__ == "__main__":
    unittest.main()
