"""Small in-memory Hub fixtures check pinning, integrity, and failure behavior."""

import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from python.prepare_llama3_model import (
    BundleError, MANIFEST, PLAN, REPO_ID, REQUIRED_TENSORS,
    audit_local_bundle, main, prepare_bundle, select_tensors, validate_config,
    write_local_audit,
)


COMMIT = "0123456789abcdef0123456789abcdef01234567"
CONFIG = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
    "max_position_embeddings": 8192, "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16", "rope_theta": 500000.0,
    "rope_scaling": None, "attention_bias": False, "mlp_bias": False,
}
ROOT = Path(__file__).resolve().parents[1]


def write_local_fixture(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    tensors = {name: torch.zeros(1, dtype=torch.bfloat16)
               for name in REQUIRED_TENSORS}
    save_file(tensors, model / "model-00001-of-00004.safetensors")
    weight_map = {name: "model-00001-of-00004.safetensors"
                  for name in REQUIRED_TENSORS}
    weight_map["model.layers.31.mlp.down_proj.weight"] = (
        "model-00004-of-00004.safetensors")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"):
        (model / name).write_text("{}", encoding="utf-8")
    return model


class FakeHub:
    def __init__(self, config=None):
        self.mapping = {name: ("embedding-part.safetensors" if name == "model.embed_tokens.weight"
                               else "layer-zero.safetensors") for name in REQUIRED_TENSORS}
        self.mapping["model.layers.0.optional_extra.weight"] = "layer-extra.safetensors"
        self.mapping["model.layers.1.self_attn.q_proj.weight"] = "unused.safetensors"
        self.files = {
            "config.json": json.dumps(CONFIG if config is None else config).encode(),
            "model.safetensors.index.json": json.dumps({"weight_map": self.mapping}).encode(),
            "tokenizer.json": b'{"fixture": "tokenizer"}',
            "tokenizer_config.json": b'{"fixture": "tokenizer_config"}',
            "special_tokens_map.json": b'{"fixture": "special_tokens"}',
            "embedding-part.safetensors": b"small embedding fixture",
            "layer-zero.safetensors": b"small layer zero fixture",
            "layer-extra.safetensors": b"small extra fixture",
            "unused.safetensors": b"layer one should not be downloaded",
        }
        self.calls = []
        self.info_calls = []
        self.corrupt = None
        self.fail = None

    def model_info(self, repo_id, **kwargs):
        self.info_calls.append((repo_id, kwargs))
        siblings = []
        for name, payload in self.files.items():
            sha = hashlib.sha256(payload).hexdigest()
            oid = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            siblings.append(SimpleNamespace(rfilename=name, size=len(payload), blob_id=oid,
                                             lfs={"sha256": sha} if name.endswith(".safetensors") else None))
        return SimpleNamespace(sha=COMMIT, siblings=siblings)

    def download(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["filename"]
        if name == self.fail:
            raise OSError("simulated interrupted download")
        target = Path(kwargs["local_dir"]) / name
        payload = self.files[name]
        if name == self.corrupt:
            payload = b"X" * len(payload)
        target.write_bytes(payload)
        return str(target)


class PrepareLlama3ModelTest(unittest.TestCase):
    def test_cli_entrypoint_runs_as_a_script(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "python" / "prepare_llama3_model.py"), "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_local_audit_separates_layer0_from_full_model_completeness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = write_local_fixture(root)
            path, report = write_local_audit(model, root / "audit")
            self.assertEqual(path.name, "local-model-audit.json")
            self.assertEqual(report["status"], "identity_unverified")
            self.assertTrue(report["layer0_assets_complete"])
            self.assertFalse(report["full_model_files_complete"])
            self.assertEqual(report["missing_full_model_shards"],
                             ["model-00004-of-00004.safetensors"])
            self.assertIsNone(report["revision"])

    def test_local_audit_rejects_truncated_layer0_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_local_fixture(Path(temp))
            shard = model / "model-00001-of-00004.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(BundleError, "不完整|大小"):
                audit_local_bundle(model)

    def test_local_audit_rejects_missing_layer0_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_local_fixture(Path(temp))
            (model / "model-00001-of-00004.safetensors").unlink()
            with self.assertRaisesRegex(BundleError, "第0层"):
                audit_local_bundle(model)

    def test_local_audit_cli_writes_report_and_rejects_remote_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = write_local_fixture(root)
            output = root / "audit"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([
                    "--local-model-dir", str(model),
                    "--output-dir", str(output),
                ]), 0)
            self.assertTrue((output / "local-model-audit.json").is_file())
            for forbidden in ("--metadata-only", "--force-download"):
                with self.subTest(forbidden=forbidden), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main([
                            "--local-model-dir", str(model),
                            "--output-dir", str(output),
                            forbidden,
                        ])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([
                        "--local-model-dir", str(model),
                        "--output-dir", str(output),
                        "--revision", "main-override",
                    ])

    def test_complete_bundle_pins_revision_selects_index_shards_and_hashes_all_files(self):
        hub = FakeHub()
        with tempfile.TemporaryDirectory() as temp:
            path, manifest = prepare_bundle(temp, "moving-branch", api=hub, downloader=hub.download)
            self.assertEqual(hub.info_calls[0][1]["revision"], "moving-branch")
            self.assertTrue(all(call["revision"] == COMMIT for call in hub.calls))
            self.assertTrue(all(call["repo_id"] == REPO_ID for call in hub.calls))
            self.assertEqual(REPO_ID, "meta-llama/Meta-Llama-3-8B")
            self.assertEqual(path.parent.name, f"Meta-Llama-3-8B-{COMMIT}")
            self.assertEqual(path.name, MANIFEST)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["repo_id"], REPO_ID)
            self.assertEqual(manifest["revision"], COMMIT)
            self.assertEqual(manifest["rope"]["rope_theta"], 500000.0)
            self.assertIsNone(manifest["rope"]["rope_scaling"])
            self.assertEqual(manifest["required_weight_shards"], ["embedding-part.safetensors",
                              "layer-extra.safetensors", "layer-zero.safetensors"])
            self.assertNotIn("unused.safetensors", manifest["files"])
            for name, record in manifest["files"].items():
                self.assertEqual(record["sha256"], hashlib.sha256(hub.files[name]).hexdigest())
                self.assertEqual(record["size_bytes"], len(hub.files[name]))
            self.assertEqual(json.loads(path.read_text()), manifest)

    def test_metadata_only_never_downloads_weights_or_claims_complete(self):
        hub = FakeHub()
        with tempfile.TemporaryDirectory() as temp:
            path, plan = prepare_bundle(temp, metadata_only=True, api=hub, downloader=hub.download)
            self.assertEqual(path.name, PLAN)
            self.assertFalse((path.parent / MANIFEST).exists())
            self.assertFalse(any(call["filename"].endswith(".safetensors") for call in hub.calls))
            self.assertEqual(plan["status"], "metadata_only_weights_pending")
            self.assertEqual(plan["files"]["layer-zero.safetensors"]["verification"], "pending")

    def test_llama3_base_config_is_accepted(self):
        validate_config(copy.deepcopy(CONFIG))

    def test_llama31_scaled_rope_is_rejected_before_other_downloads(self):
        scaled = copy.deepcopy(CONFIG)
        scaled["max_position_embeddings"] = 131072
        scaled["rope_scaling"] = {
            "rope_type": "llama3", "factor": 8.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        hub = FakeHub(scaled)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(BundleError, "unscaled"):
                prepare_bundle(temp, api=hub, downloader=hub.download)
            self.assertEqual([call["filename"] for call in hub.calls], ["config.json"])
            self.assertEqual(list(Path(temp).rglob(MANIFEST)), [])

    def test_any_non_null_rope_scaling_or_parameters_is_rejected(self):
        for field, value in (
                ("rope_scaling", {}),
                ("rope_scaling", {"rope_type": "default"}),
                ("rope_parameters", {"rope_type": "default"})):
            candidate = copy.deepcopy(CONFIG)
            candidate[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(BundleError):
                validate_config(candidate)

    def test_missing_required_tensor_and_escaping_shard_are_rejected(self):
        mapping = FakeHub().mapping
        del mapping["model.layers.0.mlp.down_proj.weight"]
        with self.assertRaises(BundleError):
            select_tensors({"weight_map": mapping})
        mapping = FakeHub().mapping
        mapping["model.embed_tokens.weight"] = "../old-weights.safetensors"
        with self.assertRaises(BundleError):
            select_tensors({"weight_map": mapping})

    def test_remote_hash_mismatch_prevents_manifest_and_removes_previous_completion(self):
        hub = FakeHub()
        with tempfile.TemporaryDirectory() as temp:
            path, _ = prepare_bundle(temp, api=hub, downloader=hub.download)
            hub.corrupt = "layer-zero.safetensors"
            with self.assertRaisesRegex(BundleError, "checksum mismatch"):
                prepare_bundle(temp, api=hub, downloader=hub.download)
            self.assertFalse(path.exists())

    def test_interrupted_download_is_resumable_without_false_completion(self):
        hub = FakeHub()
        with tempfile.TemporaryDirectory() as temp:
            hub.fail = "layer-zero.safetensors"
            with self.assertRaises(OSError):
                prepare_bundle(temp, api=hub, downloader=hub.download)
            self.assertEqual(list(Path(temp).rglob(MANIFEST)), [])
            hub.fail = None
            path, _ = prepare_bundle(temp, api=hub, downloader=hub.download)
            self.assertTrue(path.is_file())
            self.assertTrue(all(call["force_download"] is False for call in hub.calls))

    def test_unrelated_file_cannot_be_mistaken_for_this_model(self):
        hub = FakeHub()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / f"Meta-Llama-3-8B-{COMMIT}"
            folder.mkdir()
            (folder / "model.safetensors").write_bytes(b"unrelated old weights")
            with self.assertRaisesRegex(BundleError, "Unrelated file"):
                prepare_bundle(temp, api=hub, downloader=hub.download)
            self.assertFalse((folder / MANIFEST).exists())
            self.assertEqual((folder / "model.safetensors").read_bytes(), b"unrelated old weights")


if __name__ == "__main__":
    unittest.main()
