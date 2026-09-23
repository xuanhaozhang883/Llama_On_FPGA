"""Synthetic weights exercise the real safetensors loading and DDR exporter."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from python.prepare_llama3_embedding import prepare_embedding, save_embedding
from python.prepare_llama3_qkv import (bf16_le_bytes, load_saved_embedding,
                                       load_token_ids_file, main, model_metadata, project_layer0, resolve_compute,
                                       write_outputs)
from python.quantize_llama3_embedding import load_int8_rows, quantize_embedding
from python.verify_llama3_embedding import verify_embedding


class PrepareLlama3QkvTest(unittest.TestCase):
    def test_projection_rejects_empty_or_overlength_token_lists(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model"
            model.mkdir()
            config = {
                "model_type": "llama",
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
            }
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            board = {"q_heads": 2, "kv_heads": 1, "head_dim": 4, "seq_len": 4}

            for ids in ([], [0, 1, 2, 3, 4]):
                with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "1～4"):
                    project_layer0(model, ids, board)

    def test_projection_layout_padding_and_bf16(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model"
            output = Path(td) / "output"
            model.mkdir()
            cfg = {"model_type": "llama", "hidden_size": 8,
                   "num_attention_heads": 2, "num_key_value_heads": 1,
                   "head_dim": 4, "rms_norm_eps": 1e-5,
                   "rope_theta": 500000.0, "rope_scaling": {"rope_type": "llama3", "factor": 8.0}}
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
            arrays = project_layer0(model, ids, board,
                                    device="cpu", compute_dtype="fp32")
            np.testing.assert_array_equal(arrays["residual_hidden"], embedding[ids].numpy()[None, :, :])
            x = embedding[ids]
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * norm
            for name, weight, heads in (("q", q_weight, 2),
                                        ("k", k_weight, 1),
                                        ("v", v_weight, 1)):
                expected = (x @ weight.T).reshape(2, heads, 4).permute(1, 0, 2).numpy()
                np.testing.assert_allclose(arrays[name][:, :2, :], expected, rtol=1e-6)
                self.assertTrue(np.all(arrays[name][:, 2:, :] == 0))

            write_outputs(output, arrays, ids, board, model_dir=model)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["valid_tokens"], 2)
            self.assertEqual(manifest["token_ids"], ids)
            self.assertEqual(manifest["config_sha256"], hashlib.sha256((model / "config.json").read_bytes()).hexdigest())
            self.assertEqual(manifest["model_config"]["rope_scaling"], cfg["rope_scaling"])
            self.assertIsNone(manifest["weight_index_sha256"])
            self.assertNotIn("rope_parameters", manifest["model_config"])
            residual = manifest["files"]["residual_hidden"]
            self.assertEqual(residual["shape"], [1, 2, 8])
            residual_path = output / residual["file"]
            self.assertEqual(residual["sha256"], hashlib.sha256(residual_path.read_bytes()).hexdigest())
            np.testing.assert_array_equal(np.load(residual_path), embedding[ids].numpy()[None, :, :])
            for name, heads in (("q", 2), ("k", 1), ("v", 1)):
                item = manifest["files"][name]
                payload = (output / item["file"]).read_bytes()
                self.assertEqual(len(payload), heads * 4 * 4 * 2)
                self.assertEqual(item["sha256"], hashlib.sha256(payload).hexdigest())
                expected = torch.from_numpy(arrays[name]).to(torch.bfloat16)
                self.assertEqual(payload, expected.view(torch.uint16).numpy().astype("<u2").tobytes())

            # The split pipeline must produce the same residual and DDR bytes as
            # loading embeddings directly, and the CLI's default is CPU FP32.
            saved_embedding = Path(td) / "embedding"
            save_embedding(saved_embedding, model, ids, embedding[ids].numpy())
            cli_output = Path(td) / "cli"
            (Path(td) / "project_config.json").write_text(json.dumps(board), encoding="utf-8")
            with patch("python.prepare_llama3_qkv.ROOT", Path(td)):
                self.assertEqual(main(["--model-dir", str(model), "--embedding-dir", str(saved_embedding),
                                       "--output-dir", str(cli_output), "--token-ids", "3", "1"]), 0)
            cli_manifest = json.loads((cli_output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(cli_manifest["host_compute"]["device"], "cpu")
            self.assertEqual(cli_manifest["host_compute"]["dtype"], "fp32")
            self.assertEqual(cli_manifest["host_compute"]["rmsnorm_dtype"], "fp32")
            self.assertEqual(cli_manifest["host_compute"]["cpu_threads"], 1)
            self.assertTrue(cli_manifest["host_compute"]["deterministic_algorithms"])
            self.assertEqual(cli_manifest["embedding_source"]["source_weight_format"], "safetensors")
            for item in manifest["files"].values():
                self.assertEqual((cli_output / item["file"]).read_bytes(), (output / item["file"]).read_bytes())

            bad_arrays = {key: array.copy() for key, array in arrays.items()}
            bad_arrays["q"][0, -1, 0] = 1
            with self.assertRaisesRegex(ValueError, "补零"):
                write_outputs(Path(td) / "bad", bad_arrays, ids, board)

            # A partial overwrite must not retain the previous completion record.
            bad_arrays = {key: array.copy() for key, array in arrays.items()}
            bad_arrays["k"][0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "NaN"):
                write_outputs(output, bad_arrays, ids, board)
            self.assertFalse((output / "manifest.json").exists())
            with patch("python.prepare_llama3_embedding.model_metadata", side_effect=ValueError("identity failed")):
                with self.assertRaisesRegex(ValueError, "identity failed"):
                    save_embedding(saved_embedding, model, ids, embedding[ids].numpy())
            self.assertFalse((saved_embedding / "embedding_manifest.json").exists())

    def test_tie_rounds_to_even(self):
        samples = np.array([1.0 + 2**-8, 1.0 + 3 * 2**-8], dtype=np.float32)
        words = np.frombuffer(bf16_le_bytes(samples), dtype="<u2")
        np.testing.assert_array_equal(words, [0x3f80, 0x3f82])

    def test_compute_selection(self):
        self.assertEqual(resolve_compute("cpu", "auto"), ("cpu", "fp32"))
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                resolve_compute("cuda", "bf16")

    def test_embedding_stage_needs_only_its_shard(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model"
            output = Path(td) / "embedding"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "llama", "hidden_size": 8}), encoding="utf-8")
            source = torch.arange(80, dtype=torch.float32).reshape(10, 8).to(torch.bfloat16)
            save_file({"model.embed_tokens.weight": source}, model / "embedding.safetensors")
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "model.embed_tokens.weight": "embedding.safetensors",
                    "model.layers.0.input_layernorm.weight": "not-downloaded-yet.safetensors"
                }}), encoding="utf-8")
            board = {"q_heads": 2, "head_dim": 4, "seq_len": 4}
            ids = [3, 1, 3]
            embedding = prepare_embedding(model, ids, board)
            np.testing.assert_array_equal(embedding, source[ids].float().numpy())
            save_embedding(output, model, ids, embedding)
            manifest = json.loads((output / "embedding_manifest.json").read_text(encoding="utf-8"))
            index = model / "model.safetensors.index.json"
            index_bytes = index.read_bytes()
            self.assertEqual(manifest["weight_index_sha256"], hashlib.sha256(index_bytes).hexdigest())
            residual = np.load(output / "residual_hidden_fp32.npy")
            self.assertEqual(residual.shape, (1, 3, 8))
            np.testing.assert_array_equal(residual[0], embedding)
            self.assertEqual(verify_embedding(model, output), (3, 8))
            recovered = load_saved_embedding(output, model, ids, 8)
            np.testing.assert_array_equal(recovered.numpy(), embedding)
            relocated = Path(td) / "teammate-model"
            shutil.copytree(model, relocated)
            np.testing.assert_array_equal(load_saved_embedding(output, relocated, ids, 8).numpy(), embedding)
            with self.assertRaisesRegex(ValueError, "token IDs"):
                load_saved_embedding(output, model, [1, 3, 3], 8)
            index.write_bytes(index_bytes + b"\n")
            with self.assertRaisesRegex(ValueError, "权重索引"):
                load_saved_embedding(output, model, ids, 8)
            index.write_bytes(index_bytes)

            int8_dir = Path(td) / "int8"
            quantize_embedding(model, int8_dir, chunk_rows=3)
            compact = load_int8_rows(model, int8_dir, ids, 8)
            max_error = np.max(np.abs(compact - embedding), axis=1)
            row_limits = np.max(np.abs(embedding), axis=1) / 254 + 1e-6
            self.assertTrue(np.all(max_error <= row_limits))
            np.testing.assert_array_equal(
                prepare_embedding(model, ids, board, int8_dir), compact)

    def test_token_ids_file_checks_model(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            path = Path(td) / "ids.json"
            data = {"model_dir": str(model.resolve()),
                    "config_sha256": hashlib.sha256(b"{}").hexdigest(),
                    "token_ids": [128000, 123]}
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_token_ids_file(path, model), [128000, 123])
            data["token_ids"] = [True]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "无效"):
                load_token_ids_file(path, model)

    def test_token_ids_are_portable_but_tokenizer_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
            path = Path(td) / "ids.json"
            data = {**model_metadata(model), "token_ids": [3, 1]}
            path.write_text(json.dumps(data), encoding="utf-8")
            relocated = Path(td) / "teammate-model"
            shutil.copytree(model, relocated)
            self.assertEqual(load_token_ids_file(path, relocated), [3, 1])
            (relocated / "tokenizer.json").write_text('{"version": "2.0"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Tokenizer"):
                load_token_ids_file(path, relocated)

    def test_bundle_identity_and_small_file_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)
            config = model / "config.json"
            config.write_text("{}", encoding="utf-8")
            bundle = {"schema_version": 1, "status": "complete",
                      "repo_id": "synthetic-test", "revision": "a" * 40,
                      "required_weight_shards": [],
                      "files": {"config.json": {"sha256": hashlib.sha256(b"{}").hexdigest()}}}
            bundle_path = model / "model-bundle-manifest.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            data = {**model_metadata(model), "token_ids": [1]}
            ids_path = model / "ids.json"
            ids_path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(data["model_bundle"]["revision"], "a" * 40)
            self.assertEqual(load_token_ids_file(ids_path, model), [1])
            bundle["revision"] = "b" * 40
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bundle 身份"):
                load_token_ids_file(ids_path, model)
            config.write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config.json.*SHA256"):
                model_metadata(model)


if __name__ == "__main__":
    unittest.main()
