# Llama 3 目标模型迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将第0层PC–FPGA–PC闭环的唯一目标从`meta-llama/Llama-3.1-8B-Instruct`迁移到`meta-llama/Meta-Llama-3-8B` Base，并为本地模型和普通Llama 3 RoPE提供可重复验证证据。

**Architecture:** 保留现有张量布局、TCP协议、GQA Attention、PC后半层和FPGA RoPE数据通路，只替换目标模型身份、配置验收和RoPE初始化资产生成语义。迁移先生成独立候选LUT并对比当前ROM；已知的4个BF16字差异形成暂停门禁，未得到用户确认前不覆盖`mem/*.hex`、不重建bitstream。

**Tech Stack:** Python 3.12、PyTorch 2.4.1 CPU、NumPy 1.26.4、safetensors 0.5.3、huggingface_hub 0.36.2、`unittest`、PowerShell、Git。

**Spec:** `docs/superpowers/specs/2026-09-24-llama3-target-migration-design.md`

## Global Constraints

- 唯一正式目标是`meta-llama/Meta-Llama-3-8B` Base，不使用Instruct聊天模板。
- 当前只验证decoder第0层；batch固定为1，硬件序列长度固定为128。
- 目标结构固定为32层、hidden 4096、32个Q头、8个KV头、head_dim 128、intermediate 14336、vocab 128256。
- Llama 3 RoPE固定为`rope_theta=500000`、`max_position_embeddings=8192`，`rope_scaling`和`rope_parameters`必须缺失或为`null`。
- PC继续输出RoPE前Q/K/V；RoPE仍由FPGA执行；不得修改FPGA RoPE乘加和split-half旋转数据通路RTL。
- Q/K/V/Context布局、BF16小端格式、TCP v1、CRC、DDR地址、Cache职责和后半层公式保持不变。
- 正式Manifest必须有官方repo ID、40位revision和逐文件SHA-256；没有revision的本地报告只能标记`identity_unverified`。
- 当前本地目录`C:\lhm\2_Work\Llama3-8B`缺少第2、3分片；允许第0层测试，禁止宣称完整32层可用。
- 候选RoPE资产只能写入`out/llama3-rope-candidate`；未经用户确认不得覆盖`mem/sin_bf16.hex`或`mem/cos_bf16.hex`。
- `v31`、`v312`、`v313`是硬件版本命名，不是Llama 3.1模型命名，不得批量重命名。

## Review Focus

- `rope_scaling`为非空对象、空对象或`rope_parameters`非空时都必须拒绝；由Task 1的配置测试覆盖。
- 索引声称第0层分片存在但Safetensors文件被截断时，本地审计必须失败而不是生成完成报告；由Task 2的截断文件测试覆盖。
- 第0层所需分片齐全但其他层分片缺失时，必须区分“第0层可用”和“完整32层不可用”；由Task 2的部分分片测试覆盖。
- RoPE输出目录误指向仓库`mem`目录时必须拒绝写入；由Task 4的路径保护测试覆盖。
- 清理`llama31`模型命名时不能误删`v31/v312/v313`硬件回归脚本；由Task 5的范围测试覆盖。

---

### Task 1: 将官方模型包校验切换到Llama 3 Base

**Files:**
- Rename: `python/prepare_llama31_model.py` → `python/prepare_llama3_model.py`
- Rename: `tests/test_prepare_llama31_model.py` → `tests/test_prepare_llama3_model.py`

**Interfaces:**
- Consumes: Hugging Face仓库元数据、`config.json`、`model.safetensors.index.json`。
- Produces: `validate_config(config: Mapping[str, object]) -> None`、`select_tensors(index: Mapping[str, object]) -> dict[str, str]`、`prepare_bundle(output_dir: Path, revision: str = "main", metadata_only: bool = False, force_download: bool = False, api=None, downloader=None) -> tuple[Path, dict]`。

- [ ] **Step 1: 先重命名测试并写入Llama 3失败期望**

执行：

```powershell
git mv tests/test_prepare_llama31_model.py tests/test_prepare_llama3_model.py
```

把导入和fixture改为：

```python
from python.prepare_llama3_model import (
    BundleError, MANIFEST, PLAN, REPO_ID, REQUIRED_TENSORS,
    prepare_bundle, select_tensors, validate_config,
)

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
```

将原来“旧Llama 3被拒绝”的测试改成下面三项：

```python
def test_llama3_base_config_is_accepted(self):
    validate_config(copy.deepcopy(CONFIG))

def test_llama31_scaled_rope_is_rejected(self):
    scaled = copy.deepcopy(CONFIG)
    scaled["max_position_embeddings"] = 131072
    scaled["rope_scaling"] = {
        "rope_type": "llama3", "factor": 8.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    with self.assertRaisesRegex(BundleError, "unscaled"):
        validate_config(scaled)

def test_any_non_null_rope_scaling_or_parameters_is_rejected(self):
    for field, value in (
        ("rope_scaling", {}),
        ("rope_scaling", {"rope_type": "default"}),
        ("rope_parameters", {"rope_type": "default"}),
    ):
        candidate = copy.deepcopy(CONFIG)
        candidate[field] = value
        with self.subTest(field=field, value=value), self.assertRaises(BundleError):
            validate_config(candidate)
```

- [ ] **Step 2: 运行新测试并确认因模块尚未重命名而失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prepare_llama3_model -v
```

Expected: FAIL，报错包含`No module named 'python.prepare_llama3_model'`。

- [ ] **Step 3: 重命名实现并替换目标常量与配置校验**

执行：

```powershell
git mv python/prepare_llama31_model.py python/prepare_llama3_model.py
```

实现中的目标常量和目录名固定为：

```python
REPO_ID = "meta-llama/Meta-Llama-3-8B"
BUNDLE_PREFIX = "Meta-Llama-3-8B"
```

`validate_config()`使用以下目标字段，并明确拒绝scaled-RoPE：

```python
expected = {
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-5,
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
if config.get("rope_scaling") is not None or config.get("rope_parameters") is not None:
    raise BundleError("Target Llama 3 requires unscaled RoPE")
```

bundle目录改为：

```python
bundle = Path(output_dir).resolve() / f"{BUNDLE_PREFIX}-{commit}"
```

保留原有revision固定、远端哈希校验、路径逃逸保护、原子Manifest和失败撤销语义。

- [ ] **Step 4: 更新远端fixture断言并运行单文件测试**

测试必须断言：

```python
self.assertEqual(REPO_ID, "meta-llama/Meta-Llama-3-8B")
self.assertEqual(path.parent.name, f"Meta-Llama-3-8B-{COMMIT}")
self.assertEqual(manifest["repo_id"], REPO_ID)
self.assertEqual(manifest["revision"], COMMIT)
self.assertEqual(manifest["rope"]["rope_theta"], 500000.0)
self.assertIsNone(manifest["rope"]["rope_scaling"])
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prepare_llama3_model -v
```

Expected: PASS，所有Hub fixture都不访问真实网络。

- [ ] **Step 5: 确认旧模型模块名已从Python源码和测试中消失**

Run:

```powershell
rg -n "prepare_llama31|PrepareLlama31|Llama-3.1-8B-Instruct" python tests
```

Expected: 无输出；退出码1表示没有匹配项。

- [ ] **Step 6: 提交目标模型校验迁移**

```powershell
git add python/prepare_llama3_model.py tests/test_prepare_llama3_model.py
git commit -m "Migrate model bundle validation to Llama 3"
```

### Task 2: 为队友拷贝的本地模型生成诚实的第0层审计报告

**Files:**
- Modify: `python/prepare_llama3_model.py`
- Modify: `tests/test_prepare_llama3_model.py`

**Interfaces:**
- Consumes: `model_dir: Path`，其中包含原始配置、索引和部分或全部Safetensors分片。
- Produces: `audit_local_bundle(model_dir: Path) -> dict`、`write_local_audit(model_dir: Path, output_dir: Path) -> tuple[Path, dict]`，以及`local-model-audit.json`。

- [ ] **Step 1: 写本地第0层齐全、完整32层不齐全的失败测试**

用`safetensors.torch.save_file`创建包含`REQUIRED_TENSORS`键的小分片，并让索引额外引用一个不存在的后续层分片：

```python
def test_local_audit_separates_layer0_from_full_model_completeness(self):
    with tempfile.TemporaryDirectory() as temp:
        model = Path(temp) / "model"
        out = Path(temp) / "audit"
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

        path, report = write_local_audit(model, out)
        self.assertEqual(path.name, "local-model-audit.json")
        self.assertEqual(report["status"], "identity_unverified")
        self.assertTrue(report["layer0_assets_complete"])
        self.assertFalse(report["full_model_files_complete"])
        self.assertEqual(report["missing_full_model_shards"],
                         ["model-00004-of-00004.safetensors"])
        self.assertIsNone(report["revision"])
```

- [ ] **Step 2: 写截断分片和缺少第0层分片的失败测试**

```python
def test_local_audit_rejects_truncated_layer0_shard(self):
    model = self.make_local_fixture()
    shard = model / "model-00001-of-00004.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with self.assertRaisesRegex(BundleError, "不完整|大小"):
        audit_local_bundle(model)

def test_local_audit_rejects_missing_layer0_shard(self):
    model = self.make_local_fixture()
    (model / "model-00001-of-00004.safetensors").unlink()
    with self.assertRaisesRegex(BundleError, "第0层"):
        audit_local_bundle(model)
```

- [ ] **Step 3: 运行测试并确认接口尚不存在**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prepare_llama3_model -v
```

Expected: FAIL，报错指向`audit_local_bundle`或`write_local_audit`尚未定义。

- [ ] **Step 4: 实现本地审计和原子报告写入**

核心返回结构固定为：

```python
report = {
    "schema_version": 1,
    "status": "identity_unverified",
    "repo_id": REPO_ID,
    "revision": None,
    "scope": "Embedding and decoder layer 0 local smoke validation",
    "layer0_assets_complete": True,
    "full_model_files_complete": not missing_full_shards,
    "required_layer0_shards": required_layer0_shards,
    "all_index_shards": all_index_shards,
    "missing_full_model_shards": missing_full_shards,
    "tensor_to_shard": tensors,
    "config": config,
    "files": file_records,
}
```

实现规则：

```python
config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
validate_config(config)
index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
tensors = select_tensors(index)
required_layer0_shards = sorted(set(tensors.values()))
all_index_shards = sorted(set(index["weight_map"].values()))
missing_full_shards = [name for name in all_index_shards
                       if not (model_dir / name).is_file()]
for shard_name in required_layer0_shards:
    shard_path = model_dir / shard_name
    if not shard_path.is_file():
        raise BundleError(f"第0层所需分片缺失: {shard_name}")
    try:
        with SafeTensorReader(shard_path) as reader:
            for tensor_name, mapped_shard in tensors.items():
                if mapped_shard == shard_name:
                    reader.shape(tensor_name)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise BundleError(
            f"第0层分片不完整或格式错误: {shard_name}") from error
```

对`config.json`、索引、Tokenizer文件和第0层所需分片流式计算`size_bytes`与`sha256`。报告只能写入`output_dir/local-model-audit.json`，使用`write_json_atomic()`，不得写入或修改模型目录。

- [ ] **Step 5: 增加本地审计CLI分支并测试参数互斥**

CLI固定用法：

```powershell
.\.venv\Scripts\python.exe python\prepare_llama3_model.py `
  --local-model-dir "C:\lhm\2_Work\Llama3-8B" `
  --output-dir "out\llama3-model-audit"
```

本地模式禁止同时使用`--metadata-only`、`--force-download`或非默认`--revision`；测试应确认非法组合返回退出码2或抛出`SystemExit(2)`。

- [ ] **Step 6: 运行模型校验测试并提交**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prepare_llama3_model -v
git add python/prepare_llama3_model.py tests/test_prepare_llama3_model.py
git commit -m "Audit local Llama 3 layer-zero assets"
```

Expected: 测试PASS；测试不读取真实的16 GB模型。

### Task 3: 清除PC前半程fixture中的Llama 3.1 RoPE语义

**Files:**
- Modify: `python/prepare_llama3_qkv.py`
- Modify: `tests/test_prepare_llama3_qkv.py`

**Interfaces:**
- Consumes: 已通过`validate_config()`或本地审计的Llama 3目录。
- Produces: Q/K/V与Residual的Manifest显式记录普通Llama 3 RoPE字段。

- [ ] **Step 1: 把QKV测试配置改为普通Llama 3并增加显式字段断言**

```python
cfg = {
    "model_type": "llama",
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": None,
}
```

Manifest断言改为：

```python
self.assertEqual(manifest["model_config"]["rope_theta"], 500000.0)
self.assertIsNone(manifest["model_config"]["rope_scaling"])
self.assertIsNone(manifest["model_config"]["rope_parameters"])
```

再增加一个无`rope_scaling`键的配置，要求输出仍显式记录两个缩放字段为`None`。

- [ ] **Step 2: 运行QKV测试并确认缺失字段断言失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prepare_llama3_qkv -v
```

Expected: FAIL，`model_config`缺少`rope_parameters`或`rope_scaling`键。

- [ ] **Step 3: 让模型元数据稳定记录RoPE字段**

把配置摘要构造固定为：

```python
model_config_fields = (
    "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "rms_norm_eps", "torch_dtype", "dtype",
    "rope_theta", "rope_scaling", "rope_parameters",
)
metadata["model_config"] = {name: config.get(name)
                            for name in model_config_fields}
```

不要在`project_layer0()`中加入4096维目标硬编码；该函数继续支持小型合成fixture。正式模型身份由Task 1/2的校验入口保证。

- [ ] **Step 4: 运行QKV、Attention和后半层回归**

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_prepare_llama3_qkv `
  tests.test_llama_attention_reference `
  tests.test_llama_post_layer -v
```

Expected: PASS。

- [ ] **Step 5: 提交PC语义清理**

```powershell
git add python/prepare_llama3_qkv.py tests/test_prepare_llama3_qkv.py
git commit -m "Record unscaled Llama 3 RoPE metadata"
```

### Task 4: 生成普通Llama 3 RoPE候选LUT并证据化4字差异

**Files:**
- Create: `python/generate_llama3_rope_lut.py`
- Create: `tests/test_llama3_rope_lut.py`
- Create at execution time: `out/llama3-rope-candidate/sin_bf16.hex`
- Create at execution time: `out/llama3-rope-candidate/cos_bf16.hex`
- Create at execution time: `out/llama3-rope-candidate/manifest.json`
- Create at execution time: `reports/llama3_rope_migration/rom-comparison.json`

**Interfaces:**
- Consumes: 已验证的Llama 3 `config.json`和可选的现有ROM目录。
- Produces: `generate_rope_words(config: Mapping[str, object], seq_len: int = 128, head_dim: int = 128) -> tuple[np.ndarray, np.ndarray]`、`write_lut_bundle(config_path: Path, output_dir: Path, forbidden_rom_dir: Path) -> Path`、`compare_roms(candidate_dir: Path, current_rom_dir: Path, report_path: Path) -> dict`。

- [ ] **Step 1: 写公式、顺序和BF16 RNE的失败测试**

```python
def torch_reference(config, seq_len=128, head_dim=128):
    positions = torch.arange(seq_len, dtype=torch.float32)
    dims = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (float(config["rope_theta"]) ** (dims / head_dim))
    angles = torch.outer(positions, inv_freq)
    sine = angles.sin().to(torch.bfloat16).view(torch.uint16).numpy()
    cosine = angles.cos().to(torch.bfloat16).view(torch.uint16).numpy()
    return sine, cosine

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
    np.testing.assert_array_equal(cosine[0], np.full(64, 0x3F80, dtype=np.uint16))
```

- [ ] **Step 2: 写路径保护、配置拒绝和差异定位测试**

```python
def test_writer_refuses_repository_mem_directory(self):
    with self.assertRaisesRegex(ValueError, "mem"):
        write_lut_bundle(CONFIG, ROOT / "mem", forbidden_rom_dir=ROOT / "mem")

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
        "table": "cosine", "flat_index": 1,
        "position": 0, "frequency_index": 1,
        "current_word": "3F80", "candidate_word": "3F81",
    })
```

- [ ] **Step 3: 运行新测试并确认模块尚不存在**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_llama3_rope_lut -v
```

Expected: FAIL，报错包含`No module named 'python.generate_llama3_rope_lut'`。

- [ ] **Step 4: 实现HF/PyTorch顺序的普通RoPE生成器**

核心计算必须使用PyTorch FP32运算顺序：

```python
def generate_rope_words(config, seq_len=128, head_dim=128):
    validate_config(config)
    positions = torch.arange(seq_len, dtype=torch.float32)
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (float(config["rope_theta"]) ** (dimensions / head_dim))
    angles = torch.outer(positions, inv_freq)
    sine = angles.sin().to(torch.bfloat16).view(torch.uint16).numpy().copy()
    cosine = angles.cos().to(torch.bfloat16).view(torch.uint16).numpy().copy()
    return sine, cosine
```

HEX按`[position][frequency]`行优先展开，每行一个4位大写十六进制BF16字：

```python
payload = "".join(f"{int(word):04X}\n" for word in words.reshape(-1))
```

先在输出目录写`.tmp`文件，三份资产全部成功后再原子替换。`manifest.json`记录repo ID、配置SHA-256、`seq_len=128`、`head_dim=128`、元素数8192、两个HEX文件SHA-256和生成算法`torch_fp32_then_bf16_rne`。

- [ ] **Step 5: 实现只读ROM对比和JSON报告**

对每个差异记录：表名、平坦索引、position、frequency index、当前字和候选字。汇总字段固定为：

```python
{
    "candidate_manifest_sha256": candidate_manifest_sha256,
    "current_rom": {
        "sin_sha256": current_sin_sha256,
        "cos_sha256": current_cos_sha256,
    },
    "counts": {"sin": 0, "cos": 0, "total": 0},
    "mismatches": [],
    "decision": "match" if not mismatches else "user_confirmation_required",
}
```

上述摘要均由实际文件计算。另用NumPy FP64角度路径生成一份仅用于诊断、不写入HEX的参考；报告为每个差异增加`current_matches_numpy_fp64`，并汇总：

```python
"precision_diagnosis": (
    "current_rom_matches_numpy_fp64"
    if all(item["current_matches_numpy_fp64"] for item in mismatches)
    else "not_explained_by_numpy_fp64"
)
```

这样可以用证据判断4个字是否来自FP32与FP64计算顺序。报告写入`reports/llama3_rope_migration/rom-comparison.json`；比较函数只读`mem`。

- [ ] **Step 6: 运行RoPE单元测试并提交生成器**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_llama3_rope_lut -v
git add python/generate_llama3_rope_lut.py tests/test_llama3_rope_lut.py
git commit -m "Generate Llama 3 RoPE candidate LUTs"
```

Expected: 单元测试PASS；`mem/*.hex`的Git状态不变。

### Task 5: 更新现行文档并隔离历史Llama 3.1计划

**Files:**
- Modify: `docs/model_bundle.md`
- Modify: `docs/layer0_handoff_contract.md`
- Rename: `docs/superpowers/plans/2026-09-23-llama31-layer0-single-owner.md` → `docs/superpowers/plans/2026-09-23-layer0-single-owner-superseded.md`
- Modify: `docs/superpowers/plans/2026-09-23-layer0-single-owner-superseded.md`
- Modify: `docs/superpowers/specs/2026-09-23-layer0-single-owner-design.md`
- Create: `tests/test_llama3_target_docs.py`

**Interfaces:**
- Consumes: Task 1/2/4确定的脚本名、命令和状态语义。
- Produces: 唯一有效目标文档和可机器检查的命名边界。

- [ ] **Step 1: 写现行文档与文件名范围测试**

```python
class Llama3TargetDocsTest(unittest.TestCase):
    def test_active_docs_name_only_llama3_base_target(self):
        for relative in ("docs/model_bundle.md",
                         "docs/superpowers/specs/2026-09-24-llama3-target-migration-design.md",
                         "docs/superpowers/plans/2026-09-24-llama3-target-migration.md"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("meta-llama/Meta-Llama-3-8B", text)
        model_bundle = (ROOT / "docs/model_bundle.md").read_text(encoding="utf-8")
        self.assertNotIn("meta-llama/Llama-3.1-8B-Instruct", model_bundle)

    def test_model_source_and_test_filenames_do_not_use_llama31(self):
        offenders = [path for folder in (ROOT / "python", ROOT / "tests")
                     for path in folder.glob("*llama31*")]
        self.assertEqual(offenders, [])

    def test_hardware_versioned_regressions_are_preserved(self):
        for relative in ("tests/run_v31_board_log_checks.ps1",
                         "tests/run_v312_qk_multilane_checks.ps1",
                         "tests/run_v313_qk4_system_checks.ps1"):
            self.assertTrue((ROOT / relative).is_file())
```

- [ ] **Step 2: 运行文档测试并确认旧目标仍导致失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_llama3_target_docs -v
```

Expected: FAIL，至少报告旧`prepare_llama31`文件名或现行文档中的3.1目标。

- [ ] **Step 3: 重写模型交付文档**

`docs/model_bundle.md`必须给出两条可直接运行的命令：

```powershell
# 当前朋友拷贝目录：只做本地第0层审计
.\.venv\Scripts\python.exe python\prepare_llama3_model.py `
  --local-model-dir "C:\lhm\2_Work\Llama3-8B" `
  --output-dir "out\llama3-model-audit"

# 将来有官方访问权限时：固定revision并完成正式交付
.\.venv\Scripts\python.exe python\prepare_llama3_model.py `
  --output-dir "out\models" `
  --revision main
```

文档明确：本地审计的`identity_unverified`不是官方身份签核；第0层所需张量在第1分片不代表完整32层可用；Base模型不需要chat template。

- [ ] **Step 4: 标记旧交接/计划为已取代并保留审计历史**

`docs/layer0_handoff_contract.md`首段增加：

```markdown
> **已被取代：** 本文记录2026-09-22的旧双人分工和Llama 3.1目标，不再作为当前实施依据。当前模型目标以`docs/superpowers/specs/2026-09-24-llama3-target-migration-design.md`为准，当前工作由同一负责人完成，队友只提供模型文件。
```

旧实施计划重命名，并在首段增加相同的“历史文档/不再执行”说明。旧设计文档保留原文件名但增加已取代说明，避免把历史叙述误当成当前事实。

- [ ] **Step 5: 运行文档边界测试和全文检索**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_llama3_target_docs -v
rg -n "prepare_llama31|test_prepare_llama31" python tests docs\model_bundle.md
```

Expected: 测试PASS；`rg`无输出。`tests/run_v31*`、`tests/run_v312*`和`tests/run_v313*`仍存在。

- [ ] **Step 6: 提交文档迁移**

```powershell
git add docs tests/test_llama3_target_docs.py
git commit -m "Document Llama 3 layer-zero target"
```

### Task 6: 对真实本地模型执行第0层冒烟测试并停在ROM决策门禁

**Files:**
- Create at execution time: `out/llama3-model-audit/local-model-audit.json`
- Create at execution time: `out/llama3-layer0-smoke/front/*`
- Create at execution time: `out/llama3-layer0-smoke/post-summary.json`
- Create at execution time: `out/llama3-rope-candidate/*`
- Create at execution time: `reports/llama3_rope_migration/rom-comparison.json`
- Modify only if evidence is suitable for source control: `reports/llama3_rope_migration/rom-comparison.json`

**Interfaces:**
- Consumes: `C:\lhm\2_Work\Llama3-8B`、Tasks 1–5的工具和未修改的当前ROM。
- Produces: 真实第0层资产审计、前半程输出、后半程有限值检查和ROM差异决策。

- [ ] **Step 1: 运行全部Python回归建立迁移后软件基线**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Expected: 全部PASS；测试数不少于迁移前40项。

- [ ] **Step 2: 审计真实本地模型并核对状态边界**

```powershell
.\.venv\Scripts\python.exe python\prepare_llama3_model.py `
  --local-model-dir "C:\lhm\2_Work\Llama3-8B" `
  --output-dir "out\llama3-model-audit"
```

读取报告并断言：

```powershell
$audit = Get-Content "out\llama3-model-audit\local-model-audit.json" -Raw | ConvertFrom-Json
if ($audit.status -ne "identity_unverified") { throw "本地模型不能标记为正式已验证" }
if (-not $audit.layer0_assets_complete) { throw "第0层资产不完整" }
if ($audit.full_model_files_complete) { throw "缺少第2、3分片却被误报为完整模型" }
```

Expected: 第0层通过；完整模型为false；缺失列表包含`model-00002-of-00004.safetensors`和`model-00003-of-00004.safetensors`。

- [ ] **Step 3: 用一个有效Token运行真实PC前半程**

```powershell
.\.venv\Scripts\python.exe python\prepare_llama3_qkv.py `
  --model-dir "C:\lhm\2_Work\Llama3-8B" `
  --output-dir "out\llama3-layer0-smoke\front" `
  --token-ids 128000 `
  --device cpu `
  --compute-dtype fp32 `
  --num-threads 1
```

Expected: 生成固定S=128的Q/K/V BF16文件、`[1,1,4096]`的`residual_hidden_fp32.npy`和Manifest；Q/K/V补零区全零且所有输出无NaN/Inf。

- [ ] **Step 4: 用零Context运行真实PC后半程权重冒烟**

执行下面的只读脚本；它只加载第0层后半程五个权重，不加载完整8B模型：

```powershell
@'
import json
from pathlib import Path
import numpy as np
import torch
from python.llama_post_layer import load_post_weights, run_post_attention

model = Path(r"C:\lhm\2_Work\Llama3-8B")
front = Path(r"out\llama3-layer0-smoke\front")
residual = torch.from_numpy(np.load(front / "residual_hidden_fp32.npy")).float()
weights = load_post_weights(model, layer_idx=0)
result = run_post_attention(torch.zeros_like(residual), residual, weights,
                            rms_eps=1e-5)
summary = {}
for name in ("o_proj", "residual_1", "post_norm", "swiglu",
             "down_proj", "layer_output"):
    tensor = getattr(result, name)
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN/Inf")
    summary[name] = {"shape": list(tensor.shape),
                     "max_abs": float(tensor.abs().max())}
target = Path(r"out\llama3-layer0-smoke\post-summary.json")
target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(target)
'@ | .\.venv\Scripts\python.exe -
```

Expected: 六个检查点均为`[1,1,4096]`且无NaN/Inf。

- [ ] **Step 5: 生成候选LUT并只读比较当前ROM**

```powershell
.\.venv\Scripts\python.exe python\generate_llama3_rope_lut.py `
  --model-dir "C:\lhm\2_Work\Llama3-8B" `
  --output-dir "out\llama3-rope-candidate" `
  --compare-rom-dir "mem" `
  --comparison-report "reports\llama3_rope_migration\rom-comparison.json"
```

然后验证ROM没有被写入：

```powershell
git diff --exit-code -- mem\sin_bf16.hex mem\cos_bf16.hex
```

Expected: 候选各8192项；比较报告预计为sin 3项、cos 1项、合计4项差异；`decision`为`user_confirmation_required`；`mem`无diff。

- [ ] **Step 6: 在决策门禁处停止并向用户报告4项差异**

如果`counts.total`不为0，停止执行，不改ROM、不改Golden、不运行Vivado/Vitis/实板测试。向用户提供每个差异的position、frequency index、当前BF16字和候选BF16字，并说明它们是否来自FP32与FP64计算顺序差异。等待用户在以下两者中选择：

1. 采用HF/PyTorch FP32语义并另行制定ROM/Golden/bitstream更新步骤；
2. 保留当前ROM并把现有数值语义作为硬件兼容约束，同时记录其偏离HF参考。

如果`counts.total`意外为0，则只提交对比报告和测试证据，不需要ROM修改。

- [ ] **Step 7: 保存迁移软件证据并提交**

只有在报告不包含机器私密信息且`git diff --check`通过时，提交ROM比较报告：

```powershell
git add reports/llama3_rope_migration/rom-comparison.json
git diff --cached --check
git commit -m "Record Llama 3 RoPE ROM comparison"
```

`out/`继续由`.gitignore`排除；不得提交模型权重、Q/K/V、Residual或候选大张量。

## 最终验证清单

- [ ] `git status --short`只显示预期内容或为空。
- [ ] `python`与`tests`中没有`llama31`模型文件名或导入。
- [ ] 所有Python单元测试PASS且不少于迁移前40项。
- [ ] 本地报告为`identity_unverified`，第0层可用、完整32层不可用。
- [ ] 真实第0层前半程和后半程冒烟输出无NaN/Inf。
- [ ] 候选LUT与独立PyTorch参考16,384项逐字一致。
- [ ] 当前ROM的差异数量和位置已经写入报告。
- [ ] `mem/*.hex`、RTL、Golden、TCP协议和DDR配置在用户作出ROM决策前没有变化。
