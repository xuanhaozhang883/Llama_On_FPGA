# 第0层PC Attention客户端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可靠的PC端TCP v1客户端，在纯PC环境中验证固定Q/K/V请求、1 MiB Context响应、顺序连接复用以及全部错误边界。

**Architecture:** 保留`attention_protocol.py`作为纯字节协议层，新建`fpga_attention_client.py`负责前半层产物验证、Socket收发和事务状态。测试代码使用本机Loopback模拟服务器代替板卡，正式客户端只返回经过身份、长度和CRC验证的原始BF16 Context，不运行Attention或PC后半层。

**Tech Stack:** Python 3.12、标准库`socket/dataclasses/hashlib/json/zlib`、NumPy 1.26.4、`unittest`、Git。

**Spec:** `docs/superpowers/specs/2026-09-24-layer0-pc-client-design.md`

## Global Constraints

- TCP v1端口默认5001，请求头40字节，响应头32字节，所有多字节整数为无符号小端。
- 请求载荷固定为Q 1,048,576字节、K 262,144字节、V 262,144字节，顺序为`Q || K || V`。
- 成功响应固定携带1,048,576字节、BF16小端、`[head][token][dim]`布局的Context。
- 同一连接可顺序执行多笔请求；前一响应完整结束后才能发送下一请求；任何时刻最多一笔请求在途。
- 第一版不支持线程安全、并发、流水请求、自动重试、自动重连或部分Context返回。
- 第一次Socket I/O开始后的任意事务错误都关闭当前连接；本地发送前校验失败保持连接可用。
- 模拟服务器只存在于测试文件中，不进入`python/`正式模块。
- 本阶段不修改RTL、RoPE ROM、Golden、PS/Vitis工程、模型权重、Echo程序或PC后半层。
- 每个功能任务必须遵循RED→GREEN：先写失败测试、确认失败原因，再写最小实现。

## Review Focus

- Manifest和CRC完全匹配但`valid_tokens`之后存在非零Q/K/V时，加载必须在任何Socket写入前失败；Task 2覆盖。
- 请求头已经发送后Q/K/V发送超时或失败时，连接必须关闭且不能自动重发；Task 4覆盖。
- 错误响应允许`request_id/valid_tokens`为0或当前值，非零错配必须作为响应串线关闭连接；Task 3覆盖。
- `OK`响应带非零`detail_code`必须按协议错误拒绝，不能生成Context；Task 1和Task 3覆盖。
- 响应被逐字节分片、确定性不规则分片或头部与Context合包时结果必须一致，同一连接10笔请求不得交叉；Task 5覆盖。

---

## 文件结构

阶段2只修改或新增以下文件：

```text
docs/protocol/attention_tcp_v1.md
docs/superpowers/specs/2026-09-24-layer0-pc-client-design.md
docs/superpowers/plans/2026-09-24-layer0-pc-client.md
python/attention_protocol.py
python/fpga_attention_client.py
tests/test_attention_protocol.py
tests/test_fpga_attention_client.py
```

职责固定为：

- `attention_tcp_v1.md`：唯一字节级协议与连接复用语义；
- `layer0-pc-client-design.md`和本计划：已经确认的设计依据与执行记录，不在功能任务中改写需求；
- `attention_protocol.py`：无Socket依赖的头部、固定长度、状态码和CRC规则；
- `fpga_attention_client.py`：输入装载、底层收发、异常和`AttentionClient`状态；
- `test_attention_protocol.py`：协议字段和固定测试向量；
- `test_fpga_attention_client.py`：文件夹具、Socket替身和Loopback模拟服务器。

## 执行前基线

实施开始时先使用`superpowers:using-git-worktrees`从包含本计划的提交创建隔离worktree。在该worktree中确认存在可用的64位Python 3.12虚拟环境；若`.venv`不存在，执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 safetensors==0.5.3 huggingface_hub==0.36.2
```

然后建立实施前证据：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
git status --short
```

Expected: 当前57项测试全部PASS；工作区为空。若环境无法安装或基线测试失败，停止实施并先报告，不在红色基线上开发。

---

### Task 1: 冻结连接复用语义并收紧成功响应字段

**Files:**
- Modify: `docs/protocol/attention_tcp_v1.md:5-13`
- Modify: `python/attention_protocol.py:200-221`
- Modify: `tests/test_attention_protocol.py:105-160`

**Interfaces:**
- Consumes: 现有`ResponseHeader`、`StatusCode`和`_validate_response_fields(header)`。
- Produces: `OK`响应必须满足`detail_code == 0`；唯一协议文档明确顺序复用规则。

- [ ] **Step 1: 为非零成功详情码写失败测试**

在`AttentionProtocolTest.test_response_rejects_invalid_fields`末尾加入：

```python
with self.assertRaises(ProtocolError) as raised:
    pack_response_header(replace(good, detail_code=1))
self.assertEqual(raised.exception.status_code, StatusCode.BAD_HEADER)
```

- [ ] **Step 2: 运行协议测试并确认正确失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_attention_protocol -v
```

Expected: 新断言FAIL，因为当前`_validate_response_fields()`接受`OK + detail_code=1`；其他协议测试仍通过。

- [ ] **Step 3: 在协议层实现最小字段约束**

在`_validate_response_fields()`的`status is StatusCode.OK`分支加入：

```python
if header.detail_code != 0:
    raise ProtocolError(StatusCode.BAD_HEADER, "成功响应的detail_code必须为0")
```

不要对`fpga_status`增加位语义校验；具体DONE/ERROR位仍由后续PS/PL联调负责。

- [ ] **Step 4: 更新唯一协议文档的连接规则**

将第1节连接规则明确写成：

```markdown
- 同一TCP连接可顺序执行多笔请求和响应。
- 前一笔响应的响应头及可选Context完整结束后，才能发送下一笔请求。
- v1只允许一个客户端，任何时刻最多一笔请求在途；不支持流水或并发请求。
```

保持错误响应不携带Context、CRC覆盖范围和固定字节布局不变。

- [ ] **Step 5: 运行协议测试和差异检查**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_attention_protocol -v
git diff --check
```

Expected: 协议测试全部PASS；`git diff --check`无输出。

- [ ] **Step 6: 提交协议收紧**

```powershell
git add docs/protocol/attention_tcp_v1.md python/attention_protocol.py tests/test_attention_protocol.py
git commit -m "Tighten Attention response protocol"
```

---

### Task 2: 实现前半层产物的离线装载和严格验证

**Files:**
- Create: `python/fpga_attention_client.py`
- Create: `tests/test_fpga_attention_client.py`

**Interfaces:**
- Consumes: `Q_BYTES`、`K_BYTES`、`V_BYTES`、`MAX_VALID_TOKENS`、`crc32()`，以及`prepare_llama3_qkv.py`生成的Manifest v2。
- Produces: `PreparedInputs`、`AttentionClientError`、`InputValidationError`、`AttentionTransportError`、`AttentionServerError`、`load_prepared_inputs(input_dir: Path) -> PreparedInputs`。

- [ ] **Step 1: 建立确定性前半层文件夹具**

在`tests/test_fpga_attention_client.py`写入导入和辅助函数：

```python
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


def write_prepared_fixture(root: Path, valid_tokens: int = 3) -> tuple[Path, dict[str, bytes]]:
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
    shapes = {"q": [32, 128, 128], "k": [8, 128, 128], "v": [8, 128, 128]}
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
```

断言夹具本身满足固定长度：

```python
self.assertEqual(len(payloads["q"]), Q_BYTES)
self.assertEqual(len(payloads["k"]), K_BYTES)
self.assertEqual(len(payloads["v"]), V_BYTES)
```

- [ ] **Step 2: 写正常加载和数据对象失败测试**

```python
class PreparedInputsTest(unittest.TestCase):
    def test_loads_verified_manifest_and_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            output, payloads = write_prepared_fixture(Path(temp), valid_tokens=3)
            prepared = load_prepared_inputs(output)
            self.assertEqual(prepared.valid_tokens, 3)
            self.assertEqual(prepared.q_payload, payloads["q"])
            self.assertEqual(prepared.k_payload, payloads["k"])
            self.assertEqual(prepared.v_payload, payloads["v"])
            self.assertEqual(prepared.source_dir, output.resolve())
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.PreparedInputsTest.test_loads_verified_manifest_and_payloads -v
```

Expected: ERROR，模块或`load_prepared_inputs`尚不存在。

- [ ] **Step 3: 定义异常和不可变输入对象**

在`python/fpga_attention_client.py`写入：

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import socket

import numpy as np

from .attention_protocol import (
    K_BYTES, MAX_VALID_TOKENS, Q_BYTES, V_BYTES, crc32,
)


class AttentionClientError(RuntimeError):
    pass


class InputValidationError(AttentionClientError):
    pass


class AttentionTransportError(AttentionClientError):
    pass


class AttentionServerError(AttentionClientError):
    pass


@dataclass(frozen=True)
class PreparedInputs:
    valid_tokens: int
    q_payload: bytes
    k_payload: bytes
    v_payload: bytes
    source_dir: Path
```

同时定义固定元数据：

```python
_INPUT_FILES = {
    "q": ("q_before_rope_bf16.bin", (32, 128, 128), Q_BYTES),
    "k": ("k_before_rope_bf16.bin", (8, 128, 128), K_BYTES),
    "v": ("v_bf16.bin", (8, 128, 128), V_BYTES),
}
_DTYPE = "BF16 round-to-nearest-even, little-endian uint16"
_LAYOUT = "[head][token][dim] contiguous"
```

- [ ] **Step 4: 只实现能够通过正常用例的最小加载骨架**

此时只读取正常夹具需要的字段和三个文件，不提前加入严格校验；后续失败测试必须能够把缺失校验暴露为RED：

```python
def _read_manifest(input_dir: Path) -> dict:
    return json.loads(
        (input_dir / "manifest.json").read_text(encoding="utf-8"))


def load_prepared_inputs(input_dir: Path) -> PreparedInputs:
    input_dir = Path(input_dir).resolve()
    manifest = _read_manifest(input_dir)
    records = manifest["files"]
    payloads = {
        name: (input_dir / records[name]["file"]).read_bytes()
        for name in _INPUT_FILES
    }
    return PreparedInputs(
        valid_tokens=manifest["valid_tokens"],
        q_payload=payloads["q"],
        k_payload=payloads["k"],
        v_payload=payloads["v"],
        source_dir=input_dir,
    )
```

- [ ] **Step 5: 运行正常加载测试至PASS**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.PreparedInputsTest.test_loads_verified_manifest_and_payloads -v
```

Expected: PASS。

- [ ] **Step 6: 为全部Manifest失败类写测试**

增加一个辅助函数，在修改载荷后同步其Manifest摘要，仅用于构造“摘要正确但语义错误”的夹具：

```python
def rewrite_record(output: Path, manifest: dict, name: str, payload: bytes) -> None:
    filename = manifest["files"][name]["file"]
    (output / filename).write_bytes(payload)
    manifest["files"][name]["bytes"] = len(payload)
    manifest["files"][name]["crc32"] = f"{zlib.crc32(payload) & 0xffffffff:08x}"
    manifest["files"][name]["sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
```

写成可直接运行的完整测试；每个`subTest`都必须使用全新的临时目录，避免前一个变异污染后一个：

```python
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
```

- [ ] **Step 7: 运行严格输入测试并确认RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.PreparedInputsTest -v
```

Expected: 正常加载用例仍PASS；新增异常用例因最小加载骨架尚未校验Manifest身份、范围、摘要和错误转换而FAIL或ERROR。确认失败来自缺失校验，不是夹具错误。

- [ ] **Step 8: 实现Manifest和文件完整性校验**

把最小`_read_manifest()`和`load_prepared_inputs()`替换为严格版本；本步骤仍不检查补零区：

```python
def _read_manifest(input_dir: Path) -> dict:
    path = input_dir / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InputValidationError(f"无法读取有效Manifest: {path}") from error
    if not isinstance(value, dict):
        raise InputValidationError("Manifest必须是JSON对象")
    return value


def load_prepared_inputs(input_dir: Path) -> PreparedInputs:
    input_dir = Path(input_dir).resolve()
    manifest = _read_manifest(input_dir)
    if manifest.get("manifest_version") != 2:
        raise InputValidationError("只支持Manifest v2")
    valid_tokens = manifest.get("valid_tokens")
    if type(valid_tokens) is not int or not 1 <= valid_tokens <= MAX_VALID_TOKENS:
        raise InputValidationError("valid_tokens必须在1～128之间")
    if manifest.get("board_seq_len") != 128:
        raise InputValidationError("board_seq_len必须为128")
    token_ids = manifest.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != valid_tokens:
        raise InputValidationError("token_ids数量必须等于valid_tokens")
    if manifest.get("dtype") != _DTYPE or manifest.get("layout") != _LAYOUT:
        raise InputValidationError("Q/K/V数据类型或布局不符合固定接口")

    records = manifest.get("files")
    if not isinstance(records, dict):
        raise InputValidationError("Manifest缺少files对象")
    payloads = {}
    for name, (filename, shape, expected_bytes) in _INPUT_FILES.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise InputValidationError(f"Manifest缺少{name}记录")
        if record.get("file") != filename or record.get("shape") != list(shape):
            raise InputValidationError(f"{name}文件名或形状错误")
        try:
            payload = (input_dir / filename).read_bytes()
        except OSError as error:
            raise InputValidationError(f"无法读取{name}载荷: {filename}") from error
        if len(payload) != expected_bytes or record.get("bytes") != expected_bytes:
            raise InputValidationError(f"{name}长度错误")
        expected_crc = record.get("crc32")
        expected_sha = record.get("sha256")
        if (not isinstance(expected_crc, str) or
                not re.fullmatch(r"[0-9a-fA-F]{8}", expected_crc)):
            raise InputValidationError(f"{name} CRC32格式错误")
        if (not isinstance(expected_sha, str) or
                not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha)):
            raise InputValidationError(f"{name} SHA-256格式错误")
        if crc32(payload) != int(expected_crc, 16):
            raise InputValidationError(f"{name} CRC32不匹配")
        if hashlib.sha256(payload).hexdigest() != expected_sha.lower():
            raise InputValidationError(f"{name} SHA-256不匹配")
        payloads[name] = payload

    return PreparedInputs(
        valid_tokens=valid_tokens,
        q_payload=payloads["q"],
        k_payload=payloads["k"],
        v_payload=payloads["v"],
        source_dir=input_dir,
    )
```

- [ ] **Step 9: 运行严格输入测试至GREEN**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.PreparedInputsTest -v
```

Expected: 当前所有正常与异常Manifest/文件用例PASS。

- [ ] **Step 10: 写补零语义失败测试并确认RED**

```python
def test_rejects_nonzero_words_after_valid_tokens_even_with_matching_hashes(self):
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        q = bytearray((output / "q_before_rope_bf16.bin").read_bytes())
        first_invalid_word = ((0 * 128 + 3) * 128) * 2
        q[first_invalid_word:first_invalid_word + 2] = b"\x01\x00"
        rewrite_record(output, manifest, "q", bytes(q))
        with self.assertRaisesRegex(InputValidationError, "补零"):
            load_prepared_inputs(output)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.PreparedInputsTest -v
```

Expected: 补零测试FAIL，其余已实现校验PASS。

- [ ] **Step 11: 实现逐头补零检查**

增加内存对象校验函数，使Manifest加载和后续发送路径共享固定长度与补零检查：

```python
def _validate_prepared_inputs(prepared: PreparedInputs) -> None:
    if not isinstance(prepared, PreparedInputs):
        raise InputValidationError("prepared_inputs类型错误")
    if type(prepared.valid_tokens) is not int or not 1 <= prepared.valid_tokens <= 128:
        raise InputValidationError("valid_tokens必须在1～128之间")
    for name, payload in (
        ("q", prepared.q_payload),
        ("k", prepared.k_payload),
        ("v", prepared.v_payload),
    ):
        _, shape, expected_bytes = _INPUT_FILES[name]
        if not isinstance(payload, bytes) or len(payload) != expected_bytes:
            raise InputValidationError(f"{name}载荷类型或长度错误")
        words = np.frombuffer(payload, dtype="<u2").reshape(shape)
        if np.any(words[:, prepared.valid_tokens:, :] != 0):
            raise InputValidationError(
                f"{name}在valid_tokens之后不是全零补齐")
```

把`load_prepared_inputs()`结尾改为先构造、再验证：

```python
prepared = PreparedInputs(
    valid_tokens=valid_tokens,
    q_payload=payloads["q"],
    k_payload=payloads["k"],
    v_payload=payloads["v"],
    source_dir=input_dir,
)
_validate_prepared_inputs(prepared)
return prepared
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.PreparedInputsTest -v
git diff --check
```

Expected: 全部PASS；差异检查无输出。

- [ ] **Step 12: 提交输入加载器**

```powershell
git add python/fpga_attention_client.py tests/test_fpga_attention_client.py
git commit -m "Validate prepared Attention inputs"
```

---

### Task 3: 实现底层请求发送和响应接收

**Files:**
- Modify: `python/fpga_attention_client.py`
- Modify: `tests/test_fpga_attention_client.py`

**Interfaces:**
- Consumes: Task 2的`PreparedInputs`和异常类型；协议层的`RequestHeader`、`ResponseHeader`、`StatusCode`、`make_request_header()`、`pack_request_header()`、`unpack_response_header()`和`crc32()`。
- Produces: `ContextResponse`、`recv_exact(sock, size) -> bytes`、`send_qkv_request(sock, prepared_inputs, request_id) -> RequestHeader`、`recv_context_response(sock, expected_request_id, expected_valid_tokens) -> ContextResponse`。

- [ ] **Step 1: 为精确接收和EOF写失败测试**

RED阶段先只增加本步骤已经测试的导入：

```python
import socket

from python.fpga_attention_client import (
    AttentionTransportError,
    recv_exact,
)
```

该导入在`recv_exact`尚未实现时使测试模块ERROR；不要提前导入后续步骤的符号。随后在`class SocketPrimitiveTest(unittest.TestCase)`前添加Socket替身：

```python
class ChunkRecvSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.recv_calls = 0
        self.close_calls = 0

    def recv(self, size):
        self.recv_calls += 1
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) <= size:
            return chunk
        self.chunks.insert(0, chunk[size:])
        return chunk[:size]

    def close(self):
        self.close_calls += 1
```

测试：

```python
def test_recv_exact_accepts_one_byte_and_coalesced_chunks(self):
    self.assertEqual(recv_exact(ChunkRecvSocket([b"a", b"b", b"c"]), 3), b"abc")
    self.assertEqual(recv_exact(ChunkRecvSocket([b"abcdef"]), 3), b"abc")

def test_recv_exact_reports_expected_and_actual_bytes_on_eof(self):
    with self.assertRaisesRegex(
            AttentionTransportError,
            "expected_bytes=5.*actual_bytes=3"):
        recv_exact(ChunkRecvSocket([b"abc", b""]), 5)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.SocketPrimitiveTest.test_recv_exact_accepts_one_byte_and_coalesced_chunks tests.test_fpga_attention_client.SocketPrimitiveTest.test_recv_exact_reports_expected_and_actual_bytes_on_eof -v
```

Expected: ERROR，因为`recv_exact`尚不存在。

- [ ] **Step 2: 实现`recv_exact()`及异常转换**

```python
def recv_exact(sock, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise ValueError("size必须是非负整数")
    data = bytearray()
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
        except (OSError, socket.timeout) as error:
            raise AttentionTransportError(
                f"TCP接收失败：expected_bytes={size}，"
                f"actual_bytes={len(data)}") from error
        if not chunk:
            raise AttentionTransportError(
                f"连接提前关闭：expected_bytes={size}，"
                f"actual_bytes={len(data)}")
        data.extend(chunk)
    return bytes(data)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.SocketPrimitiveTest.test_recv_exact_accepts_one_byte_and_coalesced_chunks tests.test_fpga_attention_client.SocketPrimitiveTest.test_recv_exact_reports_expected_and_actual_bytes_on_eof -v
```

Expected: 两项PASS。

- [ ] **Step 3: 为请求字节顺序和发送错误写失败测试**

本步骤开始时把测试导入扩展为：

```python
from dataclasses import replace

from python.attention_protocol import (
    K_BYTES,
    Q_BYTES,
    V_BYTES,
    unpack_request_header,
    validate_request_payloads,
)
from python.fpga_attention_client import (
    AttentionTransportError,
    InputValidationError,
    recv_exact,
    send_qkv_request,
)
```

添加记录Socket：

```python
class RecordingSendSocket:
    def __init__(self, fail_on_call=None, failure=None):
        self.calls = []
        self.fail_on_call = fail_on_call
        self.failure = failure or OSError("simulated send failure")
        self.close_calls = 0

    def sendall(self, data):
        call_number = len(self.calls) + 1
        if call_number == self.fail_on_call:
            raise self.failure
        self.calls.append(bytes(data))

    def close(self):
        self.close_calls += 1
```

加入以下完整测试方法：

```python
def test_send_qkv_request_writes_exact_header_q_k_v_order(self):
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        sock = RecordingSendSocket()
        returned = send_qkv_request(sock, prepared, 0x12345678)
        self.assertEqual(len(sock.calls), 4)
        self.assertEqual(len(sock.calls[0]), 40)
        self.assertEqual(
            sock.calls[1:],
            [prepared.q_payload, prepared.k_payload, prepared.v_payload],
        )
        decoded = unpack_request_header(sock.calls[0])
        self.assertEqual(returned, decoded)
        self.assertEqual(decoded.request_id, 0x12345678)
        self.assertEqual(decoded.valid_tokens, prepared.valid_tokens)
        validate_request_payloads(decoded, *sock.calls[1:])

def test_each_send_failure_preserves_cause_and_diagnostics(self):
    labels = ("请求头", "Q", "K", "V")
    expected_bytes = (40, Q_BYTES, K_BYTES, V_BYTES)
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        for call_number, (label, size) in enumerate(
                zip(labels, expected_bytes), start=1):
            with self.subTest(label=label):
                cause = socket.timeout(f"simulated {label} timeout")
                sock = RecordingSendSocket(call_number, cause)
                with self.assertRaises(AttentionTransportError) as raised:
                    send_qkv_request(sock, prepared, 0x12345678)
                self.assertIs(raised.exception.__cause__, cause)
                message = str(raised.exception)
                self.assertIn(label, message)
                self.assertIn("request_id=305419896", message)
                self.assertIn(f"expected_bytes={size}", message)
                self.assertIn("actual_bytes=unknown", message)
                self.assertEqual(sock.close_calls, 0)

def test_send_revalidates_manual_prepared_inputs_before_writing(self):
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        invalid_length = replace(prepared, q_payload=prepared.q_payload[:-2])
        q = bytearray(prepared.q_payload)
        first_padding_word = prepared.valid_tokens * 128 * 2
        q[first_padding_word:first_padding_word + 2] = b"\x01\x00"
        invalid_padding = replace(prepared, q_payload=bytes(q))
        for label, invalid in (
            ("length", invalid_length),
            ("padding", invalid_padding),
        ):
            with self.subTest(label=label):
                sock = RecordingSendSocket()
                with self.assertRaises(InputValidationError):
                    send_qkv_request(sock, invalid, 7)
                self.assertEqual(sock.calls, [])
                self.assertEqual(sock.close_calls, 0)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.SocketPrimitiveTest.test_send_qkv_request_writes_exact_header_q_k_v_order tests.test_fpga_attention_client.SocketPrimitiveTest.test_each_send_failure_preserves_cause_and_diagnostics tests.test_fpga_attention_client.SocketPrimitiveTest.test_send_revalidates_manual_prepared_inputs_before_writing -v
```

Expected: ERROR，因为`send_qkv_request`尚不存在。

- [ ] **Step 4: 实现请求构造和发送**

把客户端模块的协议导入扩展为：

```python
from .attention_protocol import (
    K_BYTES,
    MAX_VALID_TOKENS,
    Q_BYTES,
    V_BYTES,
    ProtocolError,
    RESPONSE_HEADER_BYTES,
    RequestHeader,
    ResponseHeader,
    StatusCode,
    crc32,
    make_request_header,
    pack_request_header,
    unpack_response_header,
)
```

先增加纯本地辅助函数，供Task 4在首次I/O前调用：

```python
def _build_request(prepared_inputs: PreparedInputs, request_id: int):
    _validate_prepared_inputs(prepared_inputs)
    try:
        header = make_request_header(
            request_id=request_id,
            valid_tokens=prepared_inputs.valid_tokens,
            q_payload=prepared_inputs.q_payload,
            k_payload=prepared_inputs.k_payload,
            v_payload=prepared_inputs.v_payload,
        )
    except ValueError as error:
        raise InputValidationError("request_id或请求载荷不符合TCP v1") from error
    return header, pack_request_header(header)
```

发送辅助函数和公开函数：

```python
def _send_request_bytes(sock, header_bytes: bytes,
                        prepared_inputs: PreparedInputs,
                        request_id: int) -> None:
    for label, payload in (
        ("请求头", header_bytes),
        ("Q", prepared_inputs.q_payload),
        ("K", prepared_inputs.k_payload),
        ("V", prepared_inputs.v_payload),
    ):
        try:
            sock.sendall(payload)
        except (OSError, socket.timeout) as error:
            raise AttentionTransportError(
                f"发送{label}失败：request_id={request_id}，"
                f"expected_bytes={len(payload)}，actual_bytes=unknown") from error


def send_qkv_request(sock, prepared_inputs: PreparedInputs,
                     request_id: int) -> RequestHeader:
    header, header_bytes = _build_request(prepared_inputs, request_id)
    _send_request_bytes(
        sock, header_bytes, prepared_inputs, header.request_id)
    return header
```

导入所需协议符号。运行发送测试至PASS。

- [ ] **Step 5: 为响应分片、身份、状态、CRC和截断写失败测试**

本步骤开始时再扩展测试导入；在此之前不要导入尚未实现的响应接口：

```python
from python.attention_protocol import (
    CONTEXT_BYTES,
    ProtocolError,
    RESPONSE_HEADER_BYTES,
    ResponseHeader,
    StatusCode,
    crc32,
    pack_response_header,
)
from python.fpga_attention_client import (
    AttentionServerError,
    ContextResponse,
    recv_context_response,
)
```

在`class ResponseReceiveTest(unittest.TestCase)`前增加响应构造函数和原始头部变异函数：

```python
def response_bytes(context: bytes, request_id=7, valid_tokens=3,
                   status=StatusCode.OK, detail_code=0):
    payload = context if status is StatusCode.OK else b""
    header = ResponseHeader(
        request_id=request_id,
        status_code=status,
        valid_tokens=valid_tokens,
        context_bytes=len(payload),
        context_crc32=crc32(payload) if payload else 0,
        fpga_status=0x6D if status is StatusCode.OK else 0,
        detail_code=detail_code,
    )
    return pack_response_header(header), payload


def overwrite_header_field(raw: bytes, offset: int, replacement: bytes) -> bytes:
    changed = bytearray(raw)
    changed[offset:offset + len(replacement)] = replacement
    return bytes(changed)
```

加入以下完整测试；非法线格式必须通过修改合法头部的原始字节构造，不能调用会主动拒绝非法字段的打包函数：

```python
def test_receives_fragmented_success_response(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    header, payload = response_bytes(context)
    chunks = [bytes([value]) for value in header] + [payload[:123], payload[123:]]
    response = recv_context_response(ChunkRecvSocket(chunks), 7, 3)
    self.assertIsInstance(response, ContextResponse)
    self.assertEqual(response.context_payload, context)

def test_rejects_each_malformed_response_header_field(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    good, _ = response_bytes(context)
    cases = (
        ("magic", 0, b"NOPE"),
        ("version", 4, b"\x02"),
        ("command", 5, b"\x03"),
        ("header_bytes", 6, (31).to_bytes(2, "little")),
        ("context_bytes", 16, (CONTEXT_BYTES - 2).to_bytes(4, "little")),
    )
    for label, offset, replacement in cases:
        with self.subTest(label=label):
            sock = ChunkRecvSocket([
                overwrite_header_field(good, offset, replacement),
            ])
            with self.assertRaises(AttentionTransportError) as raised:
                recv_context_response(sock, 7, 3)
            message = str(raised.exception)
            self.assertIn("响应头", message)
            self.assertIn("request_id=7", message)
            self.assertIn("expected_bytes=32", message)
            self.assertEqual(sock.close_calls, 0)

def test_rejects_mismatched_success_identity(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    cases = (
        ("request_id", response_bytes(context, request_id=8)[0]),
        ("valid_tokens", response_bytes(context, valid_tokens=4)[0]),
    )
    for label, raw_header in cases:
        with self.subTest(label=label):
            with self.assertRaises(AttentionTransportError):
                recv_context_response(ChunkRecvSocket([raw_header]), 7, 3)

def test_maps_attributable_server_statuses_to_server_error(self):
    cases = (
        (StatusCode.BAD_CRC, 7, 3),
        (StatusCode.BUSY, 7, 3),
        (StatusCode.RUN_TIMEOUT, 7, 3),
        (StatusCode.BAD_MAGIC, 0, 0),
        (StatusCode.INTERNAL_ERROR, 0, 3),
        (StatusCode.RESET_TIMEOUT, 7, 0),
    )
    for status, request_id, valid_tokens in cases:
        with self.subTest(status=status.name):
            raw_header, _ = response_bytes(
                b"", request_id=request_id, valid_tokens=valid_tokens,
                status=status, detail_code=11)
            with self.assertRaises(AttentionServerError) as raised:
                recv_context_response(
                    ChunkRecvSocket([raw_header]), 7, 3)
            self.assertEqual(raised.exception.status_code, status)
            self.assertEqual(raised.exception.request_id, request_id)
            self.assertEqual(raised.exception.valid_tokens, valid_tokens)
            self.assertEqual(raised.exception.fpga_status, 0)
            self.assertEqual(raised.exception.detail_code, 11)

def test_rejects_nonzero_mismatched_error_identity(self):
    cases = (
        ("request_id", 8, 3),
        ("valid_tokens", 7, 4),
    )
    for label, request_id, valid_tokens in cases:
        with self.subTest(label=label):
            raw_header, _ = response_bytes(
                b"", request_id=request_id, valid_tokens=valid_tokens,
                status=StatusCode.BUSY, detail_code=1)
            with self.assertRaises(AttentionTransportError):
                recv_context_response(
                    ChunkRecvSocket([raw_header]), 7, 3)

def test_rejects_corrupt_and_truncated_context(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    raw_header, payload = response_bytes(context)
    corrupt = bytearray(payload)
    corrupt[-1] ^= 0xFF
    cases = (
        ("crc", [raw_header, bytes(corrupt)], "CRC32"),
        ("truncated", [raw_header, payload[:12345], b""], "actual_bytes=12345"),
    )
    for label, chunks, diagnostic in cases:
        with self.subTest(label=label):
            with self.assertRaises(AttentionTransportError) as raised:
                recv_context_response(ChunkRecvSocket(chunks), 7, 3)
            self.assertIn("request_id=7", str(raised.exception))
            self.assertIn(diagnostic, str(raised.exception))

def test_rejects_nonzero_detail_code_in_success_header(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    good, _ = response_bytes(context)
    invalid = overwrite_header_field(good, 28, (1).to_bytes(4, "little"))
    with self.assertRaises(AttentionTransportError) as raised:
        recv_context_response(ChunkRecvSocket([invalid]), 7, 3)
    self.assertIsInstance(raised.exception.__cause__, ProtocolError)

def test_rejects_invalid_expected_identity_before_socket_read(self):
    cases = ((-1, 3), (7, 0))
    for expected_request_id, expected_valid_tokens in cases:
        with self.subTest(
                request_id=expected_request_id,
                valid_tokens=expected_valid_tokens):
            sock = ChunkRecvSocket([])
            with self.assertRaises(InputValidationError):
                recv_context_response(
                    sock, expected_request_id, expected_valid_tokens)
            self.assertEqual(sock.recv_calls, 0)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.ResponseReceiveTest -v
```

Expected: ERROR，因为`ContextResponse`、完整`AttentionServerError`和`recv_context_response`尚不存在。

- [ ] **Step 6: 实现响应对象、服务器异常和响应接收**

先定义成功结果，并将Task 2中的空`AttentionServerError`替换为保留公开诊断字段的实现：

```python
@dataclass(frozen=True)
class ContextResponse:
    header: ResponseHeader
    context_payload: bytes


class AttentionServerError(AttentionClientError):
    def __init__(self, header: ResponseHeader):
        self.header = header
        self.status_code = header.status_code
        self.request_id = header.request_id
        self.valid_tokens = header.valid_tokens
        self.fpga_status = header.fpga_status
        self.detail_code = header.detail_code
        super().__init__(
            f"Attention服务器返回{header.status_code.name}："
            f"request_id={header.request_id}，fpga_status=0x{header.fpga_status:08x}，"
            f"detail_code={header.detail_code}")
```

然后实现接收函数：

```python
def recv_context_response(sock, expected_request_id: int,
                          expected_valid_tokens: int) -> ContextResponse:
    if type(expected_request_id) is not int or not 0 <= expected_request_id < (1 << 32):
        raise InputValidationError("expected_request_id必须是u32整数")
    if (type(expected_valid_tokens) is not int or
            not 1 <= expected_valid_tokens <= MAX_VALID_TOKENS):
        raise InputValidationError("expected_valid_tokens必须在1～128之间")
    try:
        raw_header = recv_exact(sock, RESPONSE_HEADER_BYTES)
    except AttentionTransportError as error:
        raise AttentionTransportError(
            f"接收响应头失败：request_id={expected_request_id}，"
            f"expected_bytes={RESPONSE_HEADER_BYTES}；{error}") from error
    try:
        header = unpack_response_header(raw_header)
    except ProtocolError as error:
        raise AttentionTransportError(
            f"Attention响应头不符合TCP v1：request_id={expected_request_id}，"
            f"expected_bytes={RESPONSE_HEADER_BYTES}，actual_bytes={len(raw_header)}") from error

    if header.status_code is StatusCode.OK:
        if header.request_id != expected_request_id:
            raise AttentionTransportError(
                f"成功响应request_id错误：expected={expected_request_id}，"
                f"actual={header.request_id}")
        if header.valid_tokens != expected_valid_tokens:
            raise AttentionTransportError(
                f"成功响应valid_tokens错误：request_id={expected_request_id}，"
                f"expected={expected_valid_tokens}，actual={header.valid_tokens}")
    else:
        if header.request_id not in (0, expected_request_id):
            raise AttentionTransportError(
                f"错误响应request_id串线：expected=0或{expected_request_id}，"
                f"actual={header.request_id}")
        if header.valid_tokens not in (0, expected_valid_tokens):
            raise AttentionTransportError(
                f"错误响应valid_tokens串线：request_id={expected_request_id}，"
                f"expected=0或{expected_valid_tokens}，actual={header.valid_tokens}")
        raise AttentionServerError(header)

    try:
        context = recv_exact(sock, header.context_bytes)
    except AttentionTransportError as error:
        raise AttentionTransportError(
            f"接收Context失败：request_id={expected_request_id}，"
            f"expected_bytes={header.context_bytes}；{error}") from error
    actual_crc = crc32(context)
    if actual_crc != header.context_crc32:
        raise AttentionTransportError(
            f"Context CRC32错误：request_id={expected_request_id}，"
            f"expected=0x{header.context_crc32:08x}，actual=0x{actual_crc:08x}")
    return ContextResponse(header=header, context_payload=context)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.SocketPrimitiveTest tests.test_fpga_attention_client.ResponseReceiveTest -v
```

Expected: Task 3收发测试全部PASS。

- [ ] **Step 7: 验证底层函数不擅自关闭Socket**

加入完整测试，分别触发发送失败、响应身份错误和CRC错误；这固定“底层函数不拥有Socket”的边界，Task 4再验证高层客户端会关闭：

```python
def test_low_level_failures_do_not_close_callers_socket(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)

        send_sock = RecordingSendSocket(
            fail_on_call=2, failure=OSError("simulated Q failure"))
        with self.assertRaises(AttentionTransportError):
            send_qkv_request(send_sock, prepared, 7)
        self.assertEqual(send_sock.close_calls, 0)

        mismatched_header, _ = response_bytes(context, request_id=8)
        identity_sock = ChunkRecvSocket([mismatched_header])
        with self.assertRaises(AttentionTransportError):
            recv_context_response(identity_sock, 7, 3)
        self.assertEqual(identity_sock.close_calls, 0)

        crc_header, payload = response_bytes(context)
        corrupted = bytearray(payload)
        corrupted[-1] ^= 0xFF
        crc_sock = ChunkRecvSocket([crc_header, bytes(corrupted)])
        with self.assertRaises(AttentionTransportError):
            recv_context_response(crc_sock, 7, 3)
        self.assertEqual(crc_sock.close_calls, 0)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client -v
git diff --check
```

Expected: 全部PASS；差异检查无输出。

- [ ] **Step 8: 提交底层收发**

```powershell
git add python/fpga_attention_client.py tests/test_fpga_attention_client.py
git commit -m "Implement Attention TCP transactions"
```

---

### Task 4: 实现高层客户端状态和失败即关闭策略

**Files:**
- Modify: `python/fpga_attention_client.py`
- Modify: `tests/test_fpga_attention_client.py`

**Interfaces:**
- Consumes: Task 3的`_build_request()`、`_send_request_bytes()`和`recv_context_response()`。
- Produces: `AttentionClient.connect()`、`AttentionClient.run_request()`、`AttentionClient.close()`、上下文管理器和只读`closed`属性。

- [ ] **Step 1: 写Socket生命周期替身**

把测试导入扩展为：

```python
from unittest.mock import patch

from python.fpga_attention_client import (
    AttentionClient,
    AttentionClientError,
    AttentionServerError,
    AttentionTransportError,
    InputValidationError,
)
```

```python
class ClientSocketDouble:
    def __init__(self, recv_chunks=(), fail_send_call=None,
                 send_failure=None, settimeout_failure=None,
                 close_failure=None, fail_recv_call=None,
                 recv_failure=None):
        self.recv_chunks = list(recv_chunks)
        self.sent = []
        self.send_attempts = 0
        self.recv_attempts = 0
        self.fail_send_call = fail_send_call
        self.send_failure = send_failure or socket.timeout("simulated send timeout")
        self.settimeout_failure = settimeout_failure
        self.close_failure = close_failure
        self.fail_recv_call = fail_recv_call
        self.recv_failure = recv_failure or socket.timeout("simulated recv timeout")
        self.close_calls = 0
        self.timeout = None

    def settimeout(self, value):
        if self.settimeout_failure is not None:
            raise self.settimeout_failure
        self.timeout = value

    def sendall(self, data):
        self.send_attempts += 1
        if self.send_attempts == self.fail_send_call:
            raise self.send_failure
        self.sent.append(bytes(data))

    def recv(self, size):
        self.recv_attempts += 1
        if self.recv_attempts == self.fail_recv_call:
            raise self.recv_failure
        if not self.recv_chunks:
            return b""
        chunk = self.recv_chunks.pop(0)
        if len(chunk) <= size:
            return chunk
        self.recv_chunks.insert(0, chunk[size:])
        return chunk[:size]

    def close(self):
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure
```

- [ ] **Step 2: 写成功事务、复用和幂等关闭失败测试**

新增`class AttentionClientStateTest(unittest.TestCase)`和完整成功测试：

```python
def test_two_successful_transactions_reuse_connection_and_close_is_idempotent(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        first_header, first_payload = response_bytes(context, request_id=1)
        second_header, second_payload = response_bytes(context, request_id=2)
        sock = ClientSocketDouble([
            first_header, first_payload, second_header, second_payload,
        ])
        client = AttentionClient(sock)
        first = client.run_request(prepared, request_id=1)
        second = client.run_request(prepared, request_id=2)
        self.assertEqual(first.context_payload, context)
        self.assertEqual(second.context_payload, context)
        self.assertFalse(client.closed)
        client.close()
        client.close()
        self.assertTrue(client.closed)
        self.assertEqual(sock.close_calls, 1)
```

Expected: RED，因为`AttentionClient`不存在。

- [ ] **Step 3: 写发送前失败保持连接可用测试**

```python
def test_local_validation_failure_keeps_ready_connection_usable(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        raw_header, payload = response_bytes(context, request_id=7)
        sock = ClientSocketDouble([raw_header, payload])
        client = AttentionClient(sock)
        with self.assertRaises(InputValidationError):
            client.run_request(prepared, request_id=-1)
        self.assertFalse(client.closed)
        self.assertEqual(sock.sent, [])
        response = client.run_request(prepared, request_id=7)
        self.assertEqual(response.context_payload, context)
        self.assertFalse(client.closed)
```

- [ ] **Step 4: 写发送后任意失败关闭且不重试测试**

先在测试类内增加统一断言辅助函数：

```python
def assert_transaction_failure_closes(
        self, sock, prepared, expected_exception=AttentionClientError):
    client = AttentionClient(sock)
    with self.assertRaises(expected_exception):
        client.run_request(prepared, request_id=7)
    self.assertTrue(client.closed)
    self.assertEqual(sock.close_calls, 1)
    sent_before_retry = list(sock.sent)
    with self.assertRaises(AttentionClientError):
        client.run_request(prepared, request_id=8)
    self.assertEqual(sock.sent, sent_before_retry)

def test_each_header_q_k_v_send_failure_closes_without_retry(self):
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        for call_number, label in enumerate(("header", "Q", "K", "V"), start=1):
            with self.subTest(label=label):
                sock = ClientSocketDouble(fail_send_call=call_number)
                self.assert_transaction_failure_closes(
                    sock, prepared, AttentionTransportError)

def test_receive_timeout_eof_crc_and_server_error_close_without_retry(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    good_header, good_payload = response_bytes(context)
    corrupted = bytearray(good_payload)
    corrupted[-1] ^= 0xFF
    error_header, _ = response_bytes(
        b"", status=StatusCode.BUSY, detail_code=1)
    cases = (
        ("header_timeout", ClientSocketDouble(fail_recv_call=1),
         AttentionTransportError),
        ("header_eof", ClientSocketDouble([b"short", b""]),
         AttentionTransportError),
        ("context_eof", ClientSocketDouble(
            [good_header, good_payload[:12345], b""]),
         AttentionTransportError),
        ("context_crc", ClientSocketDouble(
            [good_header, bytes(corrupted)]),
         AttentionTransportError),
        ("server_busy", ClientSocketDouble([error_header]),
         AttentionServerError),
    )
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        for label, sock, exception_type in cases:
            with self.subTest(label=label):
                self.assert_transaction_failure_closes(
                    sock, prepared, exception_type)

def test_in_flight_request_is_rejected_without_socket_io(self):
    sock = ClientSocketDouble()
    client = AttentionClient(sock)
    client._state = "IN_FLIGHT"
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        with self.assertRaisesRegex(AttentionClientError, "在途"):
            client.run_request(prepared, request_id=7)
    self.assertEqual(sock.sent, [])
    self.assertFalse(client.closed)

def test_socket_close_error_does_not_mask_primary_transaction_error(self):
    sock = ClientSocketDouble(
        fail_send_call=1,
        send_failure=socket.timeout("primary timeout"),
        close_failure=OSError("secondary close failure"),
    )
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        client = AttentionClient(sock)
        with self.assertRaises(AttentionTransportError) as raised:
            client.run_request(prepared, request_id=7)
        self.assertIsInstance(raised.exception.__cause__, socket.timeout)
        self.assertTrue(client.closed)
```

这些测试只验证顺序事务保护，不测试多线程安全。

- [ ] **Step 5: 运行核心状态测试并确认RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.AttentionClientStateTest -v
```

Expected: ERROR，因为`AttentionClient`尚不存在；确认失败来自缺失高层客户端，而不是Socket替身或响应夹具错误。

- [ ] **Step 6: 实现`AttentionClient`核心状态机**

```python
class AttentionClient:
    def __init__(self, sock):
        self._sock = sock
        self._state = "READY"

    @property
    def closed(self) -> bool:
        return self._state == "CLOSED"

    def close(self) -> None:
        if self._state != "CLOSED":
            self._state = "CLOSED"
            try:
                self._sock.close()
            except OSError:
                # Socket已经不可复用；关闭异常不能遮蔽原事务异常。
                pass

    def __enter__(self):
        if self.closed:
            raise AttentionClientError("客户端已经关闭")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def run_request(self, prepared_inputs: PreparedInputs,
                    request_id: int) -> ContextResponse:
        if self._state == "CLOSED":
            raise AttentionClientError("客户端已经关闭")
        if self._state == "IN_FLIGHT":
            raise AttentionClientError("已有Attention请求在途")

        header, header_bytes = _build_request(prepared_inputs, request_id)
        self._state = "IN_FLIGHT"
        try:
            _send_request_bytes(
                self._sock, header_bytes, prepared_inputs, header.request_id)
            response = recv_context_response(
                self._sock, header.request_id, header.valid_tokens)
        except Exception:
            self.close()
            raise
        self._state = "READY"
        return response
```

该类明确不是线程安全对象，不增加锁。

- [ ] **Step 7: 写`connect()`参数和异常转换测试**

使用`unittest.mock.patch("python.fpga_attention_client.socket.create_connection")`加入完整测试：

```python
def test_connect_sets_timeouts_and_uses_keyword_connect_timeout(self):
    sock = ClientSocketDouble()
    with patch(
            "python.fpga_attention_client.socket.create_connection",
            return_value=sock) as create_connection:
        client = AttentionClient.connect(
            "127.0.0.1", 5001, connect_timeout=1.5, io_timeout=2.5)
    create_connection.assert_called_once_with(
        ("127.0.0.1", 5001), timeout=1.5)
    self.assertEqual(sock.timeout, 2.5)
    client.close()

def test_connect_wraps_oserror_and_socket_timeout(self):
    for cause in (OSError("refused"), socket.timeout("timed out")):
        with self.subTest(cause=type(cause).__name__), patch(
                "python.fpga_attention_client.socket.create_connection",
                side_effect=cause):
            with self.assertRaises(AttentionTransportError) as raised:
                AttentionClient.connect("127.0.0.1")
            self.assertIs(raised.exception.__cause__, cause)

def test_connect_rejects_invalid_values_before_network_io(self):
    invalid_calls = (
        ("", 5001, 1.0, 1.0),
        ("127.0.0.1", 0, 1.0, 1.0),
        ("127.0.0.1", 65536, 1.0, 1.0),
        ("127.0.0.1", 5001, True, 1.0),
        ("127.0.0.1", 5001, 1.0, True),
        ("127.0.0.1", 5001, 0.0, 1.0),
        ("127.0.0.1", 5001, 1.0, float("inf")),
        ("127.0.0.1", 5001, float("nan"), 1.0),
    )
    for host, port, connect_timeout, io_timeout in invalid_calls:
        with self.subTest(
                host=host, port=port,
                connect_timeout=connect_timeout, io_timeout=io_timeout), patch(
                    "python.fpga_attention_client.socket.create_connection") as create:
            with self.assertRaises(ValueError):
                AttentionClient.connect(
                    host, port, connect_timeout=connect_timeout,
                    io_timeout=io_timeout)
            create.assert_not_called()

def test_settimeout_failure_closes_socket_and_preserves_primary_cause(self):
    primary = OSError("settimeout failed")
    sock = ClientSocketDouble(
        settimeout_failure=primary,
        close_failure=OSError("close failed"),
    )
    with patch(
            "python.fpga_attention_client.socket.create_connection",
            return_value=sock):
        with self.assertRaises(AttentionTransportError) as raised:
            AttentionClient.connect("127.0.0.1")
    self.assertIs(raised.exception.__cause__, primary)
    self.assertEqual(sock.close_calls, 1)
```

- [ ] **Step 8: 运行`connect()`测试并确认RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.AttentionClientStateTest.test_connect_sets_timeouts_and_uses_keyword_connect_timeout tests.test_fpga_attention_client.AttentionClientStateTest.test_connect_wraps_oserror_and_socket_timeout tests.test_fpga_attention_client.AttentionClientStateTest.test_connect_rejects_invalid_values_before_network_io tests.test_fpga_attention_client.AttentionClientStateTest.test_settimeout_failure_closes_socket_and_preserves_primary_cause -v
```

Expected: FAIL或ERROR，因为`AttentionClient.connect()`尚不存在。

- [ ] **Step 9: 实现`connect()`**

把下面的`classmethod`加入现有`AttentionClient`类体内；方法内容按类级缩进：

```python
@classmethod
def connect(cls, host: str, port: int = 5001, *,
            connect_timeout: float = 5.0,
            io_timeout: float = 60.0) -> "AttentionClient":
    if not isinstance(host, str) or not host:
        raise ValueError("host必须是非空字符串")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port必须在1～65535之间")
    for name, value in (("connect_timeout", connect_timeout),
                        ("io_timeout", io_timeout)):
        if (type(value) not in (int, float) or value <= 0 or
                (type(value) is float and not math.isfinite(value))):
            raise ValueError(f"{name}必须是有限正数")
    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=connect_timeout)
        sock.settimeout(io_timeout)
    except (OSError, socket.timeout) as error:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        raise AttentionTransportError(
            f"无法连接Attention服务器{host}:{port}") from error
    return cls(sock)
```

在生产模块导入`math`。关键字参数的测试与实现都固定使用`timeout=connect_timeout`，避免位置参数与关键字参数断言不一致。

- [ ] **Step 10: 运行客户端状态测试和全量客户端测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client -v
```

Expected: 全部PASS；发送失败后无自动重发，`close()`幂等，成功事务后可以顺序复用。

- [ ] **Step 11: 提交高层客户端**

```powershell
git add python/fpga_attention_client.py tests/test_fpga_attention_client.py
git diff --cached --check
git commit -m "Add reusable Attention TCP client"
```

---

### Task 5: 建立真实Loopback模拟服务器和阶段2系统门禁

**Files:**
- Modify: `tests/test_fpga_attention_client.py`

**Interfaces:**
- Consumes: Task 4完整公开客户端API和协议打包/解包函数。
- Produces: 无生产接口；提供真实TCP流下的阶段2验收证据。

- [ ] **Step 1: 实现独立的测试端精确接收函数**

把测试文件的标准库导入扩展为：

```python
import itertools
import queue
import threading
```

不要复用生产`recv_exact()`，避免客户端和测试服务器共享同一个错误。测试端使用：

```python
def server_recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise AssertionError(
                f"client closed after {len(data)} of {size} server bytes")
        data.extend(chunk)
    return bytes(data)
```

- [ ] **Step 2: 建立可传播线程异常的Loopback服务器夹具**

实现测试辅助类，固定要求：

```python
class LoopbackAttentionServer:
    def __init__(self, handler):
        self.handler = handler
        self.ready = threading.Event()
        self.errors = queue.Queue()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.settimeout(2.0)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.connection = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=2.0):
            try:
                self.listener.close()
            except OSError:
                pass
            self.thread.join(timeout=2.0)
            raise AssertionError("loopback server did not become ready")
        return self

    def _run(self):
        self.ready.set()
        try:
            self.connection, _ = self.listener.accept()
            self.connection.settimeout(2.0)
            with self.connection:
                self.handler(self.connection)
        except BaseException as error:
            self.errors.put(error)

    def __exit__(self, exc_type, exc, traceback):
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.connection.close()
            except OSError:
                pass
        try:
            self.listener.close()
        except OSError:
            pass
        self.thread.join(timeout=2.0)
        thread_error = None if self.errors.empty() else self.errors.get()
        if exc_type is None:
            if self.thread.is_alive():
                raise AssertionError("loopback server thread did not stop")
            if thread_error is not None:
                raise thread_error
        else:
            if self.thread.is_alive():
                exc.add_note("loopback server thread did not stop during cleanup")
            if thread_error is not None:
                exc.add_note(
                    "loopback server cleanup error: "
                    f"{type(thread_error).__name__}: {thread_error}")
```

测试清理不能只依赖daemon线程；每个正常结束的用例必须join成功并关闭listener和连接。测试主体已经失败时，清理和服务器线程中的次要异常不得遮蔽原始断言。

- [ ] **Step 3: 写单笔真实TCP成功测试**

把协议导入扩展为`REQUEST_HEADER_BYTES`和`unpack_request_header`，新增`class LoopbackAttentionClientTest(unittest.TestCase)`。服务器独立解析请求并逐字节验证：

```python
def test_single_request_over_real_tcp(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)

        def handler(conn):
            raw_header = server_recv_exact(conn, REQUEST_HEADER_BYTES)
            header = unpack_request_header(raw_header)
            q = server_recv_exact(conn, Q_BYTES)
            k = server_recv_exact(conn, K_BYTES)
            v = server_recv_exact(conn, V_BYTES)
            validate_request_payloads(header, q, k, v)
            self.assertEqual(
                (q, k, v),
                (prepared.q_payload, prepared.k_payload, prepared.v_payload),
            )
            response = ResponseHeader(
                request_id=header.request_id,
                status_code=StatusCode.OK,
                valid_tokens=header.valid_tokens,
                context_bytes=CONTEXT_BYTES,
                context_crc32=crc32(context),
                fpga_status=0x6D,
                detail_code=0,
            )
            conn.sendall(pack_response_header(response) + context)

        with LoopbackAttentionServer(handler) as server:
            with AttentionClient.connect(
                    "127.0.0.1", server.port,
                    connect_timeout=1.0, io_timeout=1.0) as client:
                result = client.run_request(prepared, request_id=7)
        self.assertEqual(result.context_payload, context)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fpga_attention_client.LoopbackAttentionClientTest.test_single_request_over_real_tcp -v
```

Expected: PASS，服务器线程正常结束。

- [ ] **Step 4: 写真实TCP分片和合包测试**

增加两个测试辅助函数：

```python
def send_chunks(conn, data, sizes):
    offset = 0
    for size in sizes:
        if offset >= len(data):
            break
        conn.sendall(data[offset:offset + size])
        offset += size
    if offset < len(data):
        conn.sendall(data[offset:])


def receive_and_validate_request(conn, prepared):
    header = unpack_request_header(
        server_recv_exact(conn, REQUEST_HEADER_BYTES))
    payloads = (
        server_recv_exact(conn, Q_BYTES),
        server_recv_exact(conn, K_BYTES),
        server_recv_exact(conn, V_BYTES),
    )
    validate_request_payloads(header, *payloads)
    if payloads != (
            prepared.q_payload, prepared.k_payload, prepared.v_payload):
        raise AssertionError("loopback request payload differs from fixture")
    return header


def success_wire(header, context):
    response = ResponseHeader(
        request_id=header.request_id,
        status_code=StatusCode.OK,
        valid_tokens=header.valid_tokens,
        context_bytes=CONTEXT_BYTES,
        context_crc32=crc32(context),
        fpga_status=0x6D,
        detail_code=0,
    )
    return pack_response_header(response) + context
```

使用以下完整测试覆盖合包、逐字节头部和确定性不规则分片；不要使用随机数：

```python
def test_real_tcp_accepts_coalesced_and_deterministically_chunked_responses(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    pattern = [1, 7, 64, 1023, 4096, 16384]
    irregular_sizes = list(itertools.islice(itertools.cycle(pattern), 512))
    modes = ("coalesced", "byte_header", "irregular")
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        for mode in modes:
            with self.subTest(mode=mode):
                def handler(conn, selected=mode):
                    header = receive_and_validate_request(conn, prepared)
                    wire = success_wire(header, context)
                    if selected == "coalesced":
                        conn.sendall(wire)
                    elif selected == "byte_header":
                        send_chunks(conn, wire[:RESPONSE_HEADER_BYTES], [1] * 32)
                        conn.sendall(wire[RESPONSE_HEADER_BYTES:])
                    else:
                        send_chunks(conn, wire, irregular_sizes)

                with LoopbackAttentionServer(handler) as server:
                    with AttentionClient.connect(
                            "127.0.0.1", server.port,
                            connect_timeout=1.0, io_timeout=1.0) as client:
                        result = client.run_request(prepared, request_id=7)
                self.assertEqual(result.context_payload, context)

def test_event_gated_fragment_pause_does_not_break_transaction(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    first_byte_sent = threading.Event()
    release_remaining = threading.Event()
    outcomes = queue.Queue()
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)

        def handler(conn):
            header = receive_and_validate_request(conn, prepared)
            wire = success_wire(header, context)
            conn.sendall(wire[:1])
            first_byte_sent.set()
            if not release_remaining.wait(timeout=2.0):
                raise AssertionError("test did not release fragmented response")
            conn.sendall(wire[1:])

        with LoopbackAttentionServer(handler) as server:
            client = AttentionClient.connect(
                "127.0.0.1", server.port,
                connect_timeout=1.0, io_timeout=1.0)

            def run_client():
                try:
                    outcomes.put(client.run_request(prepared, request_id=7))
                except BaseException as error:
                    outcomes.put(error)

            worker = threading.Thread(target=run_client, daemon=True)
            worker.start()
            try:
                self.assertTrue(first_byte_sent.wait(timeout=2.0))
                self.assertTrue(worker.is_alive())
                release_remaining.set()
                worker.join(timeout=2.0)
                self.assertFalse(worker.is_alive())
                outcome = outcomes.get_nowait()
                if isinstance(outcome, BaseException):
                    raise outcome
                self.assertEqual(outcome.context_payload, context)
            finally:
                release_remaining.set()
                client.close()
                worker.join(timeout=2.0)
```

- [ ] **Step 5: 写Loopback错误场景测试**

先在测试类中加入通用断言辅助函数。服务器必须先完整消费并校验请求，再发送指定原始响应：

```python
def assert_loopback_failure(
        self, prepared, raw_response, expected_exception,
        *, io_timeout=1.0):
    def handler(conn):
        receive_and_validate_request(conn, prepared)
        conn.sendall(raw_response)

    with LoopbackAttentionServer(handler) as server:
        client = AttentionClient.connect(
            "127.0.0.1", server.port,
            connect_timeout=1.0, io_timeout=io_timeout)
        with self.assertRaises(expected_exception):
            client.run_request(prepared, request_id=7)
        self.assertTrue(client.closed)
        with self.assertRaises(AttentionClientError):
            client.run_request(prepared, request_id=8)

def test_real_tcp_maps_server_statuses_and_rejects_mismatched_identity(self):
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        cases = []
        for status, request_id, valid_tokens in (
                (StatusCode.BAD_CRC, 7, 3),
                (StatusCode.BUSY, 7, 3),
                (StatusCode.RUN_TIMEOUT, 7, 3),
                (StatusCode.BAD_MAGIC, 0, 0),
                (StatusCode.INTERNAL_ERROR, 0, 3),
                (StatusCode.RESET_TIMEOUT, 7, 0)):
            raw_header, _ = response_bytes(
                b"", request_id=request_id, valid_tokens=valid_tokens,
                status=status, detail_code=1)
            cases.append((status.name, raw_header, AttentionServerError))
        mismatched, _ = response_bytes(
            b"", request_id=8, valid_tokens=3,
            status=StatusCode.BUSY, detail_code=1)
        cases.append(("mismatched_id", mismatched, AttentionTransportError))
        mismatched_tokens, _ = response_bytes(
            b"", request_id=7, valid_tokens=4,
            status=StatusCode.BUSY, detail_code=1)
        cases.append((
            "mismatched_valid_tokens",
            mismatched_tokens,
            AttentionTransportError,
        ))
        for label, raw_response, expected_exception in cases:
            with self.subTest(label=label):
                self.assert_loopback_failure(
                    prepared, raw_response, expected_exception)

def test_real_tcp_rejects_each_malformed_response_field(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    good_header, _ = response_bytes(context)
    malformed_headers = (
        ("magic", overwrite_header_field(good_header, 0, b"NOPE")),
        ("version", overwrite_header_field(good_header, 4, b"\x02")),
        ("command", overwrite_header_field(good_header, 5, b"\x03")),
        ("header_bytes", overwrite_header_field(
            good_header, 6, (31).to_bytes(2, "little"))),
        ("context_bytes", overwrite_header_field(
            good_header, 16, (CONTEXT_BYTES - 2).to_bytes(4, "little"))),
        ("detail_code", overwrite_header_field(
            good_header, 28, (1).to_bytes(4, "little"))),
    )
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        for label, raw_header in malformed_headers:
            with self.subTest(label=label):
                self.assert_loopback_failure(
                    prepared, raw_header, AttentionTransportError)

def test_real_tcp_rejects_truncated_and_corrupt_context(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    raw_header, payload = response_bytes(context)
    corrupt = bytearray(payload)
    corrupt[-1] ^= 0xFF
    cases = (
        ("truncated", raw_header + payload[:CONTEXT_BYTES // 2]),
        ("crc", raw_header + bytes(corrupt)),
    )
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)
        for label, raw_response in cases:
            with self.subTest(label=label):
                self.assert_loopback_failure(
                    prepared, raw_response, AttentionTransportError)

def test_real_tcp_no_response_times_out_without_fixed_sleep(self):
    request_received = threading.Event()
    release_server = threading.Event()
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)

        def handler(conn):
            receive_and_validate_request(conn, prepared)
            request_received.set()
            if not release_server.wait(timeout=2.0):
                raise AssertionError("test did not release no-response server")

        with LoopbackAttentionServer(handler) as server:
            client = AttentionClient.connect(
                "127.0.0.1", server.port,
                connect_timeout=1.0, io_timeout=0.3)
            try:
                with self.assertRaises(AttentionTransportError):
                    client.run_request(prepared, request_id=7)
                self.assertTrue(request_received.is_set())
                self.assertTrue(client.closed)
            finally:
                release_server.set()
                client.close()
```

这里的0.3秒只用于客户端Socket超时；服务器是否已经收全请求以及何时结束均由`threading.Event`同步，不使用固定`sleep()`。

- [ ] **Step 6: 写同一连接连续10笔请求测试**

写成完整测试；服务器在一个已接受的连接上循环10次，客户端在同一个`AttentionClient`实例上顺序调用：

```python
def test_ten_requests_reuse_one_real_tcp_connection_in_order(self):
    context = bytes(range(256)) * (CONTEXT_BYTES // 256)
    with tempfile.TemporaryDirectory() as temp:
        output, _ = write_prepared_fixture(Path(temp), valid_tokens=3)
        prepared = load_prepared_inputs(output)

        def handler(conn):
            for expected_id in range(1, 11):
                header = receive_and_validate_request(conn, prepared)
                self.assertEqual(header.request_id, expected_id)
                wire = success_wire(header, context)
                conn.sendall(wire[:RESPONSE_HEADER_BYTES])
                send_chunks(
                    conn, wire[RESPONSE_HEADER_BYTES:], [4096] * 256)

        with LoopbackAttentionServer(handler) as server:
            with AttentionClient.connect(
                    "127.0.0.1", server.port,
                    connect_timeout=1.0, io_timeout=1.0) as client:
                results = [
                    client.run_request(prepared, request_id=request_id)
                    for request_id in range(1, 11)
                ]
        self.assertEqual(len(results), 10)
        self.assertTrue(
            all(result.context_payload == context for result in results))
```

- [ ] **Step 7: 运行阶段2专属测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_attention_protocol tests.test_fpga_attention_client -v
```

Expected: 全部PASS，所有服务器线程结束，无ResourceWarning。

- [ ] **Step 8: 运行全部Python回归**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Expected: 现有57项加全部新增测试均PASS。

- [ ] **Step 9: 验证本阶段边界**

Run:

```powershell
git diff --check
git status --short
$stage2Base = git merge-base main HEAD
$allowed = @(
    "docs/protocol/attention_tcp_v1.md",
    "docs/superpowers/specs/2026-09-24-layer0-pc-client-design.md",
    "docs/superpowers/plans/2026-09-24-layer0-pc-client.md",
    "python/attention_protocol.py",
    "python/fpga_attention_client.py",
    "tests/test_attention_protocol.py",
    "tests/test_fpga_attention_client.py"
)
$changed = @(git diff --name-only "$stage2Base")
$unexpected = @($changed | Where-Object { $_ -notin $allowed })
if ($unexpected.Count -ne 0) {
    throw "阶段2出现未授权路径: $($unexpected -join ', ')"
}
$untracked = @(git ls-files --others --exclude-standard)
if ($untracked.Count -ne 0) {
    throw "阶段2出现未跟踪文件: $($untracked -join ', ')"
}
$trackedWeights = @(git ls-files "*.safetensors")
if ($trackedWeights.Count -ne 0) {
    throw "模型权重被Git跟踪: $($trackedWeights -join ', ')"
}
$trackedOutputs = @(git ls-files "out/*")
if ($trackedOutputs.Count -ne 0) {
    throw "生成输出被Git跟踪: $($trackedOutputs -join ', ')"
}
```

Expected:

- `git diff --check`无输出；
- `$changed`中的每个路径都在显式允许列表中，因此RTL、ROM、Golden、全部Vitis/PS和其他硬件资产都受到保护；
- 无未跟踪文件；
- 没有模型权重或`out/`内容进入Git。

- [ ] **Step 10: 提交Loopback验收测试**

```powershell
git add tests/test_fpga_attention_client.py
git diff --cached --check
git commit -m "Verify Attention client over loopback TCP"
```

- [ ] **Step 11: 提交后确认工作区和提交范围**

```powershell
git status --short
git log --oneline --decorate -6
```

Expected: `git status --short`无输出；最近提交依次对应协议收紧、输入加载、底层收发、高层客户端和Loopback验收，没有硬件或模型资产提交。

---

## 最终验证清单

- [ ] `docs/protocol/attention_tcp_v1.md`明确同连接顺序复用且禁止流水请求。
- [ ] `OK + detail_code != 0`被协议层拒绝。
- [ ] Manifest、Token数量、固定形状、长度、CRC、SHA-256和逐头补零全部验证。
- [ ] 输入校验失败不会向Socket写入任何字节。
- [ ] 请求流恰好为40字节头加Q、K、V，顺序和内容正确。
- [ ] 成功响应严格匹配request ID、valid tokens、Context长度和CRC。
- [ ] 合法错误头产生`AttentionServerError`，非零错配身份产生`AttentionTransportError`。
- [ ] 发送失败、接收EOF、超时和CRC错误都会关闭高层客户端且不自动重试。
- [ ] 底层函数不擅自关闭调用者Socket；`AttentionClient.close()`幂等。
- [ ] 同一真实Loopback连接连续10笔请求通过。
- [ ] 全部Python测试PASS。
- [ ] Git差异不包含RTL、ROM、Golden、PS/Vitis、Echo或模型权重。

## 实施后的下一阶段

阶段2完成后仍不代表PC–FPGA–PC闭环完成。下一份独立设计与计划将处理阶段3：PS端正式Attention服务器，包括`qkv_rx`、非阻塞PL控制、`context_tx`、中央状态机、Cache职责和Fail-stop；不会在本计划中预埋这些实现。
