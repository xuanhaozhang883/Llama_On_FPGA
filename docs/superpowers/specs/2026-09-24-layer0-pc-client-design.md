# 第0层PC Attention客户端设计

> 日期：2026-09-24
> 状态：已确认，实施计划已生成
> 适用阶段：第0层PC–FPGA–PC闭环的阶段2

## 1. 目标

阶段2实现可复用的PC端Attention TCP客户端，并在不使用开发板的条件下验证完整的请求和响应传输：

```text
PC前半层产物
→ 加载并验证Q/K/V与Manifest
→ TCP v1发送40字节请求头和固定1.5 MiB载荷
→ 本机模拟服务器返回32字节响应头和固定1 MiB Context
→ PC校验响应身份、长度和CRC
→ 输出可信的原始BF16 Context
```

本阶段证明PC客户端遵守冻结的TCP v1协议，并能正确处理TCP任意分片、错误响应、断线和超时。它不证明PS、PL、以太网或Attention数值已经通过实板验证。

## 2. 当前基线

现有代码已经提供：

- `python/attention_protocol.py`：请求/响应头、固定长度、CRC32、状态码和字段校验；
- `python/prepare_llama3_qkv.py`：生成RoPE前Q/K/V及`manifest.json`；
- `python/llama_post_layer.py`：将可信Context解码并运行PC后半层；
- `docs/protocol/attention_tcp_v1.md`：Python和C共同遵守的唯一字节级协议；
- `vitis/eth_echo`与`eth_echo_test.py`：保留为真实网络诊断工具，不纳入阶段2实现。

阶段2不重新实现协议编解码，不修改前半层和后半层数学运算，只增加输入装载、Socket收发、事务状态和本机模拟测试。

## 3. 固定设计决策

- 采用分层客户端方案，不把网络、模型计算和模拟服务器混在同一模块。
- 同一TCP连接允许顺序执行多笔请求，但任何时刻最多一笔请求在途。
- 客户端返回原始BF16 Context字节；Context解码和PC后半层由现有模块显式调用。
- 模拟服务器仅存在于测试代码中，不成为正式运行模块。
- 所有失败事务都使当前连接失效；第一版不尝试在错误后复用连接。
- 客户端不自动重试。发送中断后无法判断PS是否已经启动PL，自动重发可能重复执行或覆盖共享DDR。
- 超时由客户端参数控制，不写入TCP协议常量。
- 阶段2不提供完整端到端命令行入口；统一运行入口留到分层联调阶段。

## 4. 模块边界

### 4.1 协议层

文件：`python/attention_protocol.py`

职责保持不变：

- 定义40字节请求头和32字节响应头；
- 定义Q/K/V/Context固定长度；
- 计算CRC32；
- 打包、解包并校验协议字段。

协议层不知道文件路径、Socket连接和模型张量。

### 4.2 PC客户端层

新增：`python/fpga_attention_client.py`

职责：

- 从前半层输出目录加载并验证Q/K/V；
- 可靠发送请求头和三个载荷；
- 精确接收响应头和Context；
- 校验请求与响应的对应关系；
- 管理单请求在途和连接生命周期；
- 将错误转换为可诊断的客户端异常。

客户端层不解释Attention数值，不加载模型权重，也不运行PC后半层。

### 4.3 测试模拟服务器

新增：`tests/test_fpga_attention_client.py`

测试中的Loopback服务器监听`127.0.0.1`和系统分配的临时端口。它接收并验证请求，然后返回确定性Context，或按用例故意制造分片、错误、断线和超时。

模拟服务器不计算Attention，不监听正式端口5001，也不被生产代码导入。

## 5. 数据结构与公开接口

### 5.1 已准备输入

```python
@dataclass(frozen=True)
class PreparedInputs:
    valid_tokens: int
    q_payload: bytes
    k_payload: bytes
    v_payload: bytes
    source_dir: Path
```

加载接口：

```python
load_prepared_inputs(input_dir: Path) -> PreparedInputs
```

它读取：

```text
manifest.json
q_before_rope_bf16.bin
k_before_rope_bf16.bin
v_bf16.bin
```

并验证：

- `manifest_version == 2`；
- `valid_tokens`在1～128之间，`board_seq_len == 128`；
- `token_ids`是列表且其长度等于`valid_tokens`；
- 数据类型表示BF16 RNE小端，布局表示连续`[head][token][dim]`；
- 文件名和形状分别为Q `[32,128,128]`、K/V `[8,128,128]`；
- 实际长度分别为1,048,576、262,144、262,144字节；
- Manifest中的字节数、8位十六进制CRC32和64位十六进制SHA-256与实际文件一致；
- 对每个头，Token索引`valid_tokens`至127的所有补零字节均为零。

加载函数自身绝不联网。正式调用顺序必须先调用`load_prepared_inputs()`，再建立连接；无论调用顺序如何，本地可检测的输入错误都不得导致Socket写入。

### 5.2 底层收发接口

```python
recv_exact(sock, size: int) -> bytes

send_qkv_request(
    sock,
    prepared_inputs: PreparedInputs,
    request_id: int,
) -> RequestHeader

recv_context_response(
    sock,
    expected_request_id: int,
    expected_valid_tokens: int,
) -> ContextResponse
```

`send_qkv_request()`使用`sendall()`依次发送请求头、Q、K、V，并返回实际发送的请求头供审计。`recv_exact()`循环读取直到达到指定字节数；EOF不能被当作短消息成功。`recv_context_response()`先精确读取32字节响应头，验证后才决定是否读取Context。

因为调用者可以手工构造`PreparedInputs`而绕过Manifest加载器，发送路径必须再次验证对象类型、`valid_tokens`、三个固定载荷长度和逐头补零区；非法`request_id`也统一转换为`InputValidationError`。Manifest CRC和SHA-256只由`load_prepared_inputs()`校验，因为`PreparedInputs`不保存Manifest摘要。

三个底层函数都不拥有Socket生命周期：它们报告错误，但不直接关闭调用者传入的Socket。只有`AttentionClient`负责在事务失败时关闭连接。

### 5.3 成功响应

```python
@dataclass(frozen=True)
class ContextResponse:
    header: ResponseHeader
    context_payload: bytes
```

只有同时满足以下条件才构造`ContextResponse`：

- 响应头符合TCP v1；
- 状态为`OK`；
- `request_id`等于当前请求；
- `valid_tokens`等于当前输入；
- `context_bytes == 1_048_576`；
- `detail_code == 0`；
- 已完整接收Context；
- Context CRC32匹配响应头。

### 5.4 高层客户端

```python
class AttentionClient:
    @classmethod
    def connect(
        cls,
        host: str,
        port: int = 5001,
        *,
        connect_timeout: float = 5.0,
        io_timeout: float = 60.0,
    ) -> "AttentionClient": ...

    def run_request(
        self,
        prepared_inputs: PreparedInputs,
        request_id: int,
    ) -> ContextResponse: ...

    def close(self) -> None: ...
```

该类支持上下文管理器。调用者显式提供u32范围的`request_id`，便于测试、日志关联和后续统一运行入口管理。

`AttentionClient`不是线程安全对象；`IN_FLIGHT`只保护顺序事务生命周期，不承诺处理多线程竞争。`close()`必须幂等，上下文管理器退出后状态固定为`CLOSED`。

Socket关闭失败不能恢复连接，因此`close()`仍把客户端永久标记为`CLOSED`，并且不能让二次关闭异常遮蔽原始发送、接收或协议异常。

内部状态只有：

```text
READY → IN_FLIGHT → READY
  │          │
  └──────────┴──任一失败→ CLOSED
```

`IN_FLIGHT`时再次调用`run_request()`必须拒绝。`CLOSED`后必须创建新客户端，不能隐式重连。

## 6. 一笔事务的数据流

1. 上层先调用`load_prepared_inputs()`；该函数不创建网络副作用。
2. `AttentionClient.run_request()`检查客户端为`READY`。
3. 在第一次Socket I/O前验证`PreparedInputs`和u32 `request_id`，根据实际Q/K/V计算CRC并构造、打包完整请求头。本地校验失败时客户端保持`READY`。
4. 客户端将状态改为`IN_FLIGHT`，然后使用`sendall()`按`header → Q → K → V`发送1,572,904字节。
5. 使用`recv_exact()`接收32字节响应头。
6. 校验协议字段、状态、request ID和valid tokens。
7. 成功响应继续精确接收1,048,576字节Context并校验CRC。
8. 构造`ContextResponse`，状态回到`READY`。
9. 上层可把`context_payload`交给`decode_context_bf16()`，但该调用不属于阶段2事务。

## 7. 错误模型

```python
AttentionClientError
├── InputValidationError
├── AttentionTransportError
└── AttentionServerError
```

- `InputValidationError`：Manifest、文件、形状、长度或输入CRC错误；发生在发送前。
- `AttentionTransportError`：连接、发送、超时、EOF、响应身份、响应长度或Context CRC错误。
- `AttentionServerError`：收到格式正确且身份可归属当前事务的非`OK`响应；异常保留`status_code`、`request_id`、`fpga_status`和`detail_code`。

现有`ProtocolError`继续表达字节协议字段错误；客户端将它作为底层原因保留在`AttentionTransportError`中。

响应身份规则固定为：

- 成功响应的`request_id`和`valid_tokens`必须严格等于当前请求；
- 错误响应的`request_id`只允许为0或当前请求ID；
- 错误响应的`valid_tokens`只允许为0或当前有效Token数；
- 满足上述规则的合法错误头转换为`AttentionServerError`；
- 非零但不匹配的ID或Token数表示响应串线，转换为`AttentionTransportError`。

协议文档规定成功响应的`detail_code`必须为0。实施时在`attention_protocol.py`的响应字段校验中补齐这一约束，并增加协议回归测试。

第一次Socket写入前的本地校验失败不会影响连接。第一次Socket I/O开始后，任意异常都由`AttentionClient`关闭当前Socket并将客户端永久置为`CLOSED`。客户端不返回部分Context，不自动重试，不自动重连。

`connect()`、底层发送和接收遇到`OSError`或`socket.timeout`时统一转换为`AttentionTransportError`；协议解析错误也转换为该类型，并用`raise ... from error`保留底层原因。

错误消息必须包含操作阶段、期望字节数、实际字节数和request ID等可用信息。例如：

```text
接收Context时连接关闭：request_id=7，期望1048576字节，实际393216字节
```

## 8. 超时语义

- `connect_timeout`只用于建立连接；
- 建立成功后Socket切换为`io_timeout`；
- `sendall()`使用一个Socket超时处理整个调用；`recv_exact()`中的每次`recv()`都会重新开始Socket超时；因此`io_timeout`不是严格的整笔事务总时限，也不是跨所有分片累计的空闲计时器；
- 测试使用较短超时，实板阶段可根据PL耗时调整；
- 超时后连接关闭，不自动重发。

## 9. 测试设计

### 9.1 输入验证

覆盖正常输入，以及Manifest缺失或无效、任一载荷缺失、`token_ids`数量不一致、非法Token数、错误文件名/形状/布局、错误长度、错误CRC/SHA-256和有效前缀之后存在非零数据。断言所有输入错误发生在网络发送前。

### 9.2 底层Socket行为

覆盖一次收全、每次1字节、确定性不规则分片、合包、部分数据后EOF和超时。验证请求流恰好为：

```text
40字节头 + 1,048,576字节Q + 262,144字节K + 262,144字节V
```

测试数据使用非零确定性模式，避免顺序错误被全零载荷掩盖。

发送侧另外覆盖连接失败、连接超时、请求头发送后失败、Q/K/V中途`sendall()`失败和发送超时；断言失败后连接关闭、没有自动重发，并保留底层异常原因。

### 9.3 Loopback事务

模拟服务器覆盖：

- 正常Context响应；
- 响应头逐字节发送；
- Context确定性不规则分片；
- 响应头与部分Context合并到同一TCP流；
- 慢速分片；
- 错Magic、版本、命令、头长度、request ID和valid tokens；
- 成功响应带非零`detail_code`；
- Context长度或CRC错误；
- Context发送一部分后断线；
- 合法`BAD_CRC`、`BUSY`、`RUN_TIMEOUT`等错误响应；
- 错误响应使用允许的0或当前request ID/valid tokens，以及非法的非零错配身份；
- 不响应导致超时。

测试线程使用事件同步和系统临时端口，不依赖固定延时或正式端口，结束时必须回收线程和Socket，并把服务器线程异常传回测试主线程。

### 9.4 顺序复用

在同一TCP连接上依次执行10笔请求，request ID为1～10。每笔响应完成后再发送下一笔，验证全部Context逐字节一致。任一事务失败后，继续调用必须立即失败且不能再向旧Socket发送数据。

### 9.5 回归

完整测试命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

现有57项测试必须继续通过；新增测试不得依赖模型权重、开发板、Vivado或Vitis。

## 10. 验收标准

阶段2只有同时满足以下条件才算完成：

1. 前半层产物在连接前完成固定格式和CRC验证；
2. 请求总长度、字段和Q/K/V顺序逐字节正确；
3. 任意合法TCP分片不改变接收结果；
4. 正常1 MiB Context完整接收并通过身份、长度和CRC校验；
5. 错误响应不会产生`ContextResponse`；
6. EOF、超时和损坏Context不会返回部分数据；
7. 同一连接连续10笔请求全部成功且结果一致；
8. 失败连接不能继续复用，也没有自动重试；
9. 全部Python测试通过；
10. RTL、RoPE ROM、Golden、PS/Vitis工程和模型权重没有变化。

同时更新`docs/protocol/attention_tcp_v1.md`，把连接语义冻结为：同一连接可顺序执行多笔请求；只有前一响应完整结束后才能发送下一请求；任何时刻最多一笔请求在途。

## 11. 明确排除

阶段2不实现：

- PS端Attention服务器；
- 开发板TCP Echo或实板以太网验收；
- FPGA Attention和RoPE计算；
- PC后半层自动调用；
- 完整PC–FPGA–PC运行入口；
- 并发、流水请求和多客户端；
- 自动重试、自动重连或断点续传；
- 传输性能优化和压缩；
- 32层循环、KV Cache或文本生成。
