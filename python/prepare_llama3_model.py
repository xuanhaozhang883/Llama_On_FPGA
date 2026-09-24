"""Acquire a pinned, verified official Llama 3 Base layer-0 model bundle.

Uses huggingface_hub's existing authentication and resumable downloads. A plan
is not a complete bundle; only model-bundle-manifest.json signals completion.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "meta-llama/Meta-Llama-3-8B"
BUNDLE_PREFIX = "Meta-Llama-3-8B"
MANIFEST = "model-bundle-manifest.json"
PLAN = "bundle-plan.json"
REQUIRED_FILES = (
    "config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json",
)
OPTIONAL_FILES = ("generation_config.json", "tokenizer.model", "chat_template.jinja",
                  "LICENSE", "USE_POLICY.md", "README.md")
REQUIRED_TENSORS = {
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    *(f"model.layers.0.self_attn.{name}_proj.weight" for name in ("q", "k", "v", "o")),
    *(f"model.layers.0.mlp.{name}_proj.weight" for name in ("gate", "up", "down")),
}


class BundleError(ValueError):
    """An invalid or incomplete model bundle; safe to display to the user."""


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("rope_scaling") is not None or config.get("rope_parameters") is not None:
        raise BundleError("Target Llama 3 requires unscaled RoPE")
    expected = {
        "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
        "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 128256,
        "max_position_embeddings": 8192, "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise BundleError(f"Target config mismatch: {name} must equal {value!r}")
    if config.get("architectures") != ["LlamaForCausalLM"]:
        raise BundleError("Target config must declare LlamaForCausalLM")
    if config.get("head_dim", 128) != 128:
        raise BundleError("Target config head_dim must equal 128")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise BundleError("Target config dtype must be bfloat16")
    if config.get("attention_bias", False) or config.get("mlp_bias", False):
        raise BundleError("Target config must not use Attention or MLP bias")


def safe_repo_filename(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise BundleError("Unsafe filename in repository index")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise BundleError("Unsafe filename in repository index")
    return name


def select_tensors(index: Mapping[str, object]) -> dict[str, str]:
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict):
        raise BundleError("Weight index is missing weight_map")
    missing = REQUIRED_TENSORS - mapping.keys()
    if missing:
        raise BundleError("Weight index lacks required tensors: " + ", ".join(sorted(missing)))
    selected = {name: safe_repo_filename(shard) for name, shard in mapping.items()
                if name == "model.embed_tokens.weight" or name.startswith("model.layers.0.")}
    if any(not shard.endswith(".safetensors") for shard in selected.values()):
        raise BundleError("Only safetensors weight shards are supported")
    return dict(sorted(selected.items()))


def file_hashes(path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    return size, sha256.hexdigest(), git_blob.hexdigest()


def remote_record(sibling):
    lfs = sibling.lfs
    lfs_sha = (lfs.get("sha256") if isinstance(lfs, dict)
               else getattr(lfs, "sha256", None))
    if lfs_sha and not re.fullmatch(r"[0-9a-f]{64}", lfs_sha):
        raise BundleError("Malformed remote LFS checksum")
    return {"size_bytes": sibling.size, "lfs_sha256": lfs_sha,
            "git_blob_id": sibling.blob_id}


def verify_file(path, remote):
    size, sha256, git_blob = file_hashes(path)
    if remote["size_bytes"] is not None and size != remote["size_bytes"]:
        raise BundleError(f"Size mismatch for {path.name}; retry with --force-download")
    if remote["lfs_sha256"]:
        matches = sha256 == remote["lfs_sha256"]
    elif remote["git_blob_id"]:
        matches = git_blob == remote["git_blob_id"]
    else:
        raise BundleError(f"No remote checksum available for {path.name}")
    if not matches:
        raise BundleError(f"Remote checksum mismatch for {path.name}; retry with --force-download")
    return {"size_bytes": size, "sha256": sha256,
            "remote": remote, "verification": "verified"}


def write_json_atomic(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_bundle(output_dir, revision="main", metadata_only=False,
                   force_download=False, api=None, downloader=None):
    api = api if api is not None else HfApi()
    downloader = downloader if downloader is not None else hf_hub_download
    # Resolve once. Every following download uses the same immutable commit.
    info = api.model_info(REPO_ID, revision=revision, files_metadata=True, token=True)
    commit = info.sha
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise BundleError("Hub did not return a full 40-character commit SHA")
    if re.fullmatch(r"[0-9a-f]{40}", revision) and revision != commit:
        raise BundleError("Resolved commit differs from the requested immutable revision")
    bundle = Path(output_dir).resolve() / f"{BUNDLE_PREFIX}-{commit}"
    if bundle.is_symlink():
        raise BundleError("Bundle directory must not be a symlink")
    bundle.mkdir(parents=True, exist_ok=True)
    if bundle.resolve().parent != Path(output_dir).resolve():
        raise BundleError("Bundle directory escapes the output root")
    # A failed re-verification must not leave a stale completion assertion.
    (bundle / MANIFEST).unlink(missing_ok=True)
    (bundle / PLAN).unlink(missing_ok=True)
    siblings = {item.rfilename: item for item in info.siblings}
    missing = set(REQUIRED_FILES) - siblings.keys()
    if missing:
        raise BundleError("Official repository lacks required files: " + ", ".join(sorted(missing)))
    metadata_files = list(REQUIRED_FILES) + [name for name in OPTIONAL_FILES if name in siblings]
    records = {}

    def download_verified(name):
        safe_repo_filename(name)
        target = bundle / name
        if bundle.resolve() not in target.resolve().parents or target.is_symlink():
            raise BundleError("File path escapes the bundle directory")
        returned = Path(downloader(repo_id=REPO_ID, filename=name, revision=commit,
                                   local_dir=str(bundle), force_download=force_download, token=True))
        if returned.resolve() != target.resolve():
            raise BundleError("Downloader returned an unexpected output path")
        records[name] = verify_file(target, remote_record(siblings[name]))
        return target

    config_path = download_verified("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    index_path = download_verified("model.safetensors.index.json")
    tensors = select_tensors(json.loads(index_path.read_text(encoding="utf-8")))
    shards = sorted(set(tensors.values()))
    if any(name not in siblings for name in shards):
        raise BundleError("Weight index refers to a shard missing from the pinned repository")
    allowed = set(metadata_files + shards + [MANIFEST, PLAN, MANIFEST + ".tmp", PLAN + ".tmp"])
    for path in bundle.rglob("*"):
        relative = path.relative_to(bundle).as_posix()
        if relative == ".cache" or relative.startswith(".cache/"):
            continue
        if path.is_file() and relative not in allowed:
            raise BundleError(f"Unrelated file in bundle directory: {relative}")
    for name in metadata_files:
        if name not in records:
            download_verified(name)
    for name in shards:
        if metadata_only:
            records[name] = {"verification": "pending", "remote": remote_record(siblings[name])}
        else:
            download_verified(name)
    result = {
        "schema_version": 1,
        "status": "metadata_only_weights_pending" if metadata_only else "complete",
        "repo_id": REPO_ID, "revision": commit,
        "source": f"https://huggingface.co/{REPO_ID}/tree/{commit}",
        "scope": "Embedding and the complete decoder layer 0; not the full 32-layer model",
        "config": config,
        "rope": {key: config.get(key) for key in ("rope_theta", "rope_scaling", "rope_parameters")},
        "tensor_to_shard": tensors, "required_weight_shards": shards,
        "files": dict(sorted(records.items())),
    }
    artifact = bundle / (PLAN if metadata_only else MANIFEST)
    write_json_atomic(artifact, result)
    return artifact, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Root for a new directory named after the exact model commit")
    parser.add_argument("--revision", default="main", help="HF ref, resolved to a full commit before downloads")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Download config/index/tokenizer and write a plan; do not download weights")
    parser.add_argument("--force-download", action="store_true", help="Replace cached files from the pinned source")
    args = parser.parse_args()
    try:
        artifact, result = prepare_bundle(args.output_dir, args.revision, args.metadata_only, args.force_download)
    except BundleError as error:
        print(f"Bundle validation failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        # Avoid logging request URLs, authorization headers, or token-bearing exceptions.
        print(f"Bundle preparation failed ({type(error).__name__}). Check network access, "
              "existing Hugging Face login, and access to the official gated repository. "
              "No completed manifest was created by this attempt.", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "revision": result["revision"],
                      "artifact": str(artifact), "required_weight_shards": result["required_weight_shards"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
