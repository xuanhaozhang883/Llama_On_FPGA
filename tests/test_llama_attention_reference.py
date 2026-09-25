"""标准FP32与RTL位感知GQA Attention参考测试。"""

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from python.llama_attention_reference import (
    apply_split_half_rope,
    bf16_words_to_float,
    float_to_bf16_words,
    rtl_aware_gqa_attention,
    standard_gqa_attention,
)
from python.flash_attention_tile_model import load_lut


ROOT = Path(__file__).resolve().parents[1]


class LlamaAttentionReferenceTest(unittest.TestCase):
    def test_split_half_rope_uses_first_half_second_half_pairs(self):
        values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
        sine = torch.ones((1, 2), dtype=torch.float32)
        cosine = torch.zeros((1, 2), dtype=torch.float32)

        rotated = apply_split_half_rope(values, sine, cosine)

        torch.testing.assert_close(
            rotated,
            torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]]),
            rtol=0,
            atol=0,
        )

    def test_standard_gqa_attention_matches_independent_causal_calculation(self):
        q = torch.tensor([
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [2.0, 0.0]],
        ], dtype=torch.float32)
        k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        v = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]], dtype=torch.float32)
        sine = torch.zeros((2, 1), dtype=torch.float32)
        cosine = torch.ones((2, 1), dtype=torch.float32)

        result = standard_gqa_attention(q, k, v, sine, cosine)

        scale = 1.0 / math.sqrt(2.0)
        q0_row1_weight0 = math.exp(scale) / (math.exp(scale) + math.exp(scale))
        q1_row1_weight0 = math.exp(2.0 * scale) / (math.exp(2.0 * scale) + 1.0)
        expected = torch.tensor([
            [[2.0, 4.0],
             [q0_row1_weight0 * 2.0 + (1.0 - q0_row1_weight0) * 6.0,
              q0_row1_weight0 * 4.0 + (1.0 - q0_row1_weight0) * 8.0]],
            [[2.0, 4.0],
             [q1_row1_weight0 * 2.0 + (1.0 - q1_row1_weight0) * 6.0,
              q1_row1_weight0 * 4.0 + (1.0 - q1_row1_weight0) * 8.0]],
        ], dtype=torch.float32)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_standard_reference_rejects_invalid_shapes_and_nonfinite_values(self):
        q = torch.zeros((2, 4, 4), dtype=torch.float32)
        k = torch.zeros((1, 4, 4), dtype=torch.float32)
        v = torch.zeros_like(k)
        sine = torch.zeros((4, 2), dtype=torch.float32)
        cosine = torch.ones_like(sine)

        cases = [
            (q[:, :, :3], k, v, sine, cosine, "head_dim"),
            (q, k[:, :3, :], v, sine, cosine, "shape|形状"),
            (torch.zeros((3, 4, 4)), torch.zeros((2, 4, 4)),
             torch.zeros((2, 4, 4)), sine, cosine, "整除"),
        ]
        for args in cases:
            with self.subTest(message=args[-1]), self.assertRaisesRegex(ValueError, args[-1]):
                standard_gqa_attention(*args[:-1])

        q_bad = q.clone()
        q_bad[0, 0, 0] = math.nan
        with self.assertRaisesRegex(ValueError, "NaN|无穷"):
            standard_gqa_attention(q_bad, k, v, sine, cosine)

    def test_rtl_aware_reference_preserves_constant_v_for_causal_rows(self):
        q = np.zeros((2, 4, 4), dtype=np.float32)
        k = np.zeros((1, 4, 4), dtype=np.float32)
        constant_v = np.array([1.0, -0.5, 2.0, 0.25], dtype=np.float32)
        v = np.broadcast_to(constant_v, (1, 4, 4)).copy()
        sine = np.zeros((4, 2), dtype=np.float32)
        cosine = np.ones((4, 2), dtype=np.float32)
        lut = load_lut(ROOT / "mem" / "exp_lut_q15.mem")

        result_words = rtl_aware_gqa_attention(
            float_to_bf16_words(q),
            float_to_bf16_words(k),
            float_to_bf16_words(v),
            float_to_bf16_words(sine),
            float_to_bf16_words(cosine),
            lut,
            tile=4,
        )

        expected_row = torch.tensor(constant_v).to(torch.bfloat16).view(torch.uint16).numpy()
        expected = np.broadcast_to(expected_row, (2, 4, 4))
        np.testing.assert_array_equal(result_words, expected)
        np.testing.assert_array_equal(
            bf16_words_to_float(result_words),
            np.broadcast_to(constant_v, (2, 4, 4)),
        )

    def test_rtl_aware_reference_maps_each_q_group_to_its_kv_head(self):
        q = np.zeros((4, 4, 4), dtype=np.float32)
        k = np.zeros((2, 4, 4), dtype=np.float32)
        v = np.empty((2, 4, 4), dtype=np.float32)
        v[0] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        v[1] = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        sine = np.zeros((4, 2), dtype=np.float32)
        cosine = np.ones((4, 2), dtype=np.float32)
        lut = load_lut(ROOT / "mem" / "exp_lut_q15.mem")

        result_words = rtl_aware_gqa_attention(
            float_to_bf16_words(q),
            float_to_bf16_words(k),
            float_to_bf16_words(v),
            float_to_bf16_words(sine),
            float_to_bf16_words(cosine),
            lut,
            tile=4,
        )

        expected_kv0 = np.array([0x3F80, 0x4000, 0x4040, 0x4080], dtype=np.uint16)
        expected_kv1 = np.array([0x40A0, 0x40C0, 0x40E0, 0x4100], dtype=np.uint16)
        np.testing.assert_array_equal(
            result_words[0:2], np.broadcast_to(expected_kv0, (2, 4, 4)))
        np.testing.assert_array_equal(
            result_words[2:4], np.broadcast_to(expected_kv1, (2, 4, 4)))

    def test_rtl_aware_reference_applies_causal_mask_per_query_row(self):
        q = np.zeros((2, 4, 4), dtype=np.float32)
        k = np.zeros((1, 4, 4), dtype=np.float32)
        v = np.array([[[0.0, 2.0, 4.0, 6.0],
                       [2.0, 4.0, 6.0, 8.0],
                       [4.0, 6.0, 8.0, 10.0],
                       [6.0, 8.0, 10.0, 12.0]]], dtype=np.float32)
        sine = np.zeros((4, 2), dtype=np.float32)
        cosine = np.ones((4, 2), dtype=np.float32)
        lut = load_lut(ROOT / "mem" / "exp_lut_q15.mem")

        result_words = rtl_aware_gqa_attention(
            float_to_bf16_words(q),
            float_to_bf16_words(k),
            float_to_bf16_words(v),
            float_to_bf16_words(sine),
            float_to_bf16_words(cosine),
            lut,
            tile=4,
        )

        expected_row0 = np.array([0x0000, 0x4000, 0x4080, 0x40C0], dtype=np.uint16)
        expected_row1 = np.array([0x3F80, 0x4040, 0x40A0, 0x40E0], dtype=np.uint16)
        np.testing.assert_array_equal(
            result_words[:, 0, :], np.broadcast_to(expected_row0, (2, 4)))
        np.testing.assert_array_equal(
            result_words[:, 1, :], np.broadcast_to(expected_row1, (2, 4)))

    def test_rtl_aware_reference_rejects_bad_shapes_and_lut(self):
        q = np.zeros((2, 4, 4), dtype=np.uint16)
        k = np.zeros((1, 4, 4), dtype=np.uint16)
        v = np.zeros_like(k)
        sine = np.zeros((4, 2), dtype=np.uint16)
        cosine = np.zeros_like(sine)
        lut = [0x8000] * 513

        with self.assertRaisesRegex(ValueError, "整除"):
            rtl_aware_gqa_attention(
                np.zeros((3, 4, 4), dtype=np.uint16),
                np.zeros((2, 4, 4), dtype=np.uint16),
                np.zeros((2, 4, 4), dtype=np.uint16),
                sine,
                cosine,
                lut,
                tile=4,
            )
        with self.assertRaisesRegex(ValueError, "LUT"):
            rtl_aware_gqa_attention(q, k, v, sine, cosine, lut[:512], tile=4)
        with self.assertRaisesRegex(ValueError, "uint16"):
            rtl_aware_gqa_attention(q.astype(np.int32), k, v, sine, cosine, lut, tile=4)
        with self.assertRaisesRegex(ValueError, "tile|TILE|4"):
            rtl_aware_gqa_attention(q, k, v, sine, cosine, lut, tile=2)


if __name__ == "__main__":
    unittest.main()
