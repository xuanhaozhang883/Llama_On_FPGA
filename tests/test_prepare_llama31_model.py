"""Small in-memory Hub fixtures check pinning, integrity, and failure behavior."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from python.prepare_llama31_model import (
    BundleError, MANIFEST, PLAN, REPO_ID, REQUIRED_TENSORS,
    prepare_bundle, select_tensors, validate_config,
)


COMMIT = "0123456789abcdef0123456789abcdef01234567"
CONFIG = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
    "max_position_embeddings": 131072, "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16", "rope_theta": 500000.0,
    "rope_scaling": {"rope_type": "llama3", "factor": 8.0,
                     "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                     "original_max_position_embeddings": 8192},
}


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


class PrepareLlama31ModelTest(unittest.TestCase):
    def test_complete_bundle_pins_revision_selects_index_shards_and_hashes_all_files(self):
        hub = FakeHub()
        with tempfile.TemporaryDirectory() as temp:
            path, manifest = prepare_bundle(temp, "moving-branch", api=hub, downloader=hub.download)
            self.assertEqual(hub.info_calls[0][1]["revision"], "moving-branch")
            self.assertTrue(all(call["revision"] == COMMIT for call in hub.calls))
            self.assertTrue(all(call["repo_id"] == REPO_ID for call in hub.calls))
            self.assertEqual(path.name, MANIFEST)
            self.assertEqual(manifest["status"], "complete")
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

    def test_old_llama3_config_is_rejected_before_other_downloads(self):
        old = copy.deepcopy(CONFIG)
        old["max_position_embeddings"] = 8192
        old["rope_scaling"] = None
        hub = FakeHub(old)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(BundleError):
                prepare_bundle(temp, api=hub, downloader=hub.download)
            self.assertEqual([call["filename"] for call in hub.calls], ["config.json"])
            self.assertEqual(list(Path(temp).rglob(MANIFEST)), [])

    def test_rope_validation_including_alternate_parameters_field(self):
        altered = copy.deepcopy(CONFIG)
        altered["rope_scaling"]["high_freq_factor"] = 1.0
        with self.assertRaises(BundleError):
            validate_config(altered)
        alternate = copy.deepcopy(CONFIG)
        alternate["rope_parameters"] = alternate.pop("rope_scaling")
        alternate["rope_parameters"]["rope_theta"] = alternate.pop("rope_theta")
        validate_config(alternate)
        alternate["rope_scaling"] = {"rope_type": "default"}
        with self.assertRaises(BundleError):
            validate_config(alternate)

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
            folder = Path(temp) / f"Llama-3.1-8B-Instruct-{COMMIT}"
            folder.mkdir()
            (folder / "model.safetensors").write_bytes(b"unrelated old weights")
            with self.assertRaisesRegex(BundleError, "Unrelated file"):
                prepare_bundle(temp, api=hub, downloader=hub.download)
            self.assertFalse((folder / MANIFEST).exists())
            self.assertEqual((folder / "model.safetensors").read_bytes(), b"unrelated old weights")


if __name__ == "__main__":
    unittest.main()
