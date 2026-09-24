# Llama 3.1 8B 第 0 层 PC–FPGA–PC 单人实施计划

> **历史文档，已被取代：** 本计划不再执行。当前目标与实施步骤以`docs/superpowers/specs/2026-09-24-llama3-target-migration-design.md`和`docs/superpowers/plans/2026-09-24-llama3-target-migration.md`为准。

> 日期：2026-09-23  
> 状态：已规划，尚未开始功能实现  
> 设计依据：`docs/superpowers/specs/2026-09-23-layer0-single-owner-design.md`

## 1. 项目目标

使用官方 `meta-llama/Llama-3.1-8B-Instruct` 的真实第 0 层权重，完成下面的可复现闭环：

```text
PC（CPU FP32）
Tokenizer → Embedding → input RMSNorm → Q/K/V Projection
                         │
                         │ RoPE 前 Q/K/V，BF16/TCP
                         ▼
PS（A53 裸机 + lwIP）
接收 Q/K/V → 写入 DDR → Cache Flush → 启动并轮询 PL
                         │
                         ▼
PL
Llama 3.1 scaled RoPE → GQA Attention → Context
                         │
                         │ BF16/TCP
                         ▼
PC（CPU FP32）
Context 拼头 → o_proj → Residual → RMSNorm
→ SwiGLU MLP → Residual → 第 0 层输出
```

第一阶段只验证第 0 层，不实现32层循环、KV Cache、逐Token Decode、动态KV长度、LM Head、采样、聊天模板或GPU优化。

## 2. 责任与外部依赖

项目内全部代码、构建和联调由你负责。队友只提供目标模型文件，不再负责前半层、网络、PS服务器、PL控制、后半层或联调。

模型交付包至少包括：

- 原始 `config.json`；
- Tokenizer相关文件；
- `model.safetensors.index.json`；
- Embedding和完整第0层所需的全部权重分片；
- 模型仓库、固定revision、逐文件大小和SHA-256。

目标模型固定为：

```text
仓库：meta-llama/Llama-3.1-8B-Instruct
revision：0e9e39f249a16976918f6564b8830bc894c89659
```

模型包尚未到达时，可以完成协议、合成测试、PC后半层框架、模拟网络、PS服务器和旧硬件基线；不能替换正式RoPE ROM，也不能宣称完成目标模型验收。

## 3. 当前起点

开发工作树：

```text
C:\lhm\2_Work\Llama_On_FPGA_layer0_freeze
```

基线提交：

```text
e20d221b722be6e0b88295279041607cc7bbcfc2
```

设计提交：

```text
419a767  Document single-owner layer-zero design
```

已验证环境：

```text
Python 3.12.10
torch 2.4.1+cpu
numpy 1.26.4
safetensors 0.5.3
huggingface_hub 0.36.2
```

当前15项测试已经通过：

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_prepare_llama3_qkv `
  tests.test_prepare_llama31_model -v
```

这些测试证明前半层工具和模型包准备工具的局部行为，不证明TCP、PS、PL、RoPE或完整第0层闭环已经完成。

## 4. 固定接口

### 4.1 张量

| 张量 | 数据类型和布局 | 元素数 | 字节数 |
|---|---|---:|---:|
| Q | BF16小端 `[32,128,128]` | 524,288 | 1,048,576 |
| K | BF16小端 `[8,128,128]` | 131,072 | 262,144 |
| V | BF16小端 `[8,128,128]` | 131,072 | 262,144 |
| Context | BF16小端 `[32,128,128]` | 524,288 | 1,048,576 |
| `residual_hidden` | CPU FP32 `[1,L,4096]` | `L×4096` | 仅留在PC |

固定语义：

- `batch=1`、`layer_index=0`、硬件序列长度`S=128`；
- `valid_tokens=L`，范围`1..128`；
- 有效Token从第0行开始，后续行右补零；
- PC输出RoPE前Q/K/V，PL只执行一次RoPE；
- `residual_hidden`取自input RMSNorm之前；
- FPGA输出`o_proj`之前的完整Context。

Context进入`o_proj`前必须执行：

```python
context_heads = context_words.reshape(32, 128, 128)
context = context_heads.permute(1, 0, 2).contiguous()
context = context.reshape(1, 128, 4096)
context = context[:, :valid_tokens, :]
```

### 4.2 DDR与Cache

| 张量 | 起始地址 | 长度 |
|---|---:|---:|
| Q | `0x10000000` | `0x100000` |
| K | `0x10100000` | `0x040000` |
| V | `0x10140000` | `0x040000` |
| Context | `0x10180000` | `0x100000` |

保留区间为：

```text
[0x10000000, 0x10280000)
```

职责固定为：

- `qkv_rx`只负责接收、写入DDR和计算输入CRC；
- `fpt_attention_hw`负责Q/K/V Flush、Context清零及必要Flush、Reset、Start和Poll；
- `context_tx`在PL成功后对Context执行Invalidate，然后计算Context CRC并发送；
- `attention_server`负责事务状态和模块调度，不重复执行Cache操作。

### 4.3 PL成功条件

成功时FPGA状态必须同时包含：

```text
DONE | V_LOADED | CORE_DONE | CONTEXT_DONE
```

并且不能包含：

```text
BUSY | ERROR
```

具体位定义以现有RTL和板端测试代码核对后的公共头文件为准，不能凭名称重新猜测位号。

### 4.4 TCP v1

固定规则：

- TCP端口`5001`；
- 同一连接完成请求和响应；
- 单客户端、单请求在途；
- 所有多字节整数小端；
- CRC为`zlib.crc32(payload) & 0xffffffff`；
- 禁止直接发送带有潜在填充字节的C结构体。

请求头固定40字节：

| 偏移 | 字段 | 类型或固定值 |
|---:|---|---|
| 0 | magic | `4s="FPTA"` |
| 4 | version | `u8=1` |
| 5 | command | `u8=1`，`RUN_ATTENTION` |
| 6 | header_bytes | `u16=40` |
| 8 | request_id | `u32` |
| 12 | valid_tokens | `u16`，`1..128` |
| 14 | layer_index | `u16=0` |
| 16/20/24 | Q/K/V字节数 | 三个`u32` |
| 28/32/36 | Q/K/V CRC32 | 三个`u32` |

请求载荷依次为：

```text
Q || K || V
```

响应头固定32字节：

| 偏移 | 字段 | 类型或固定值 |
|---:|---|---|
| 0 | magic | `4s="FPTA"` |
| 4 | version | `u8=1` |
| 5 | command | `u8=2`，`ATTENTION_RESULT` |
| 6 | header_bytes | `u16=32` |
| 8 | request_id | `u32` |
| 12 | status_code | `u16` |
| 14 | valid_tokens | `u16` |
| 16 | context_bytes | 成功时`1,048,576` |
| 20 | context_crc32 | `u32` |
| 24 | fpga_status | `u32` |
| 28 | detail_code | `u32` |

错误响应只有32字节响应头，`context_bytes=0`、`context_crc32=0`。

状态码固定为：

| 值 | 名称 |
|---:|---|
| 0 | `OK` |
| 1 | `BAD_MAGIC` |
| 2 | `BAD_VERSION` |
| 3 | `BAD_HEADER` |
| 4 | `BAD_LENGTH` |
| 5 | `BAD_VALID_TOKENS` |
| 6 | `BAD_LAYER_INDEX` |
| 7 | `BAD_CRC` |
| 8 | `BUSY` |
| 9 | `RESET_TIMEOUT` |
| 10 | `RUN_TIMEOUT` |
| 11 | `FPGA_ERROR` |
| 12 | `INTERNAL_ERROR` |

### 4.5 Fail-stop

- 能证明PL尚未启动的协议错误，可以返回错误头；只有完整消费当前帧且帧边界明确时才允许复用连接，否则关闭连接。
- Start已经发出，或无法排除已经发出后发生超时、硬件错误，服务器进入`FAULT`。
- `FAULT`下禁止新请求和共享DDR覆盖，断线重连不能清除故障。
- 只有经过验证的系统复位，并确认PL/AXI不再访问共享DDR后，才能恢复。
- `READY=1`、`BUSY=0`或软复位完成不能单独证明AXI事务已经排空。

## 5. 文件结构

计划新增或修改：

```text
docs/protocol/attention_tcp_v1.md
docs/runbooks/layer0_hybrid_bringup.md

python/attention_protocol.py
python/fpga_attention_client.py
python/llama_attention_reference.py
python/llama_post_layer.py
python/generate_llama31_rope_lut.py
python/run_layer0_hybrid.py
python/prepare_llama3_qkv.py                 # 仅在复核发现问题时修改
python/prepare_llama31_model.py              # 仅在交付检查需要时修改

tests/test_attention_protocol.py
tests/test_fpga_attention_client.py
tests/test_llama_attention_reference.py
tests/test_llama_post_layer.py
tests/test_llama31_rope_lut.py
tests/test_run_layer0_hybrid.py

vitis/attention_server/src/attention_protocol.h
vitis/attention_server/src/qkv_rx.c
vitis/attention_server/src/qkv_rx.h
vitis/attention_server/src/fpt_attention_hw.c
vitis/attention_server/src/fpt_attention_hw.h
vitis/attention_server/src/context_tx.c
vitis/attention_server/src/context_tx.h
vitis/attention_server/src/attention_server.c
vitis/attention_server/src/attention_server.h
vitis/attention_server/src/main.c
```

原`vitis/src/fpt_attention_board_test.c`继续作为独立硬件回归，不把网络服务逻辑直接塞入其中。

## 6. 实施阶段

工作包编号表达技术依赖，不要求所有编号完全串行。模型包未到时，优先推进不依赖正式模型的任务。

### 阶段A：基线和协议冻结

#### A1：整理开发基线

涉及内容：

- 保留当前`.venv`，确认`.gitignore`忽略`/.venv/`、`out/`和模型权重；
- 新建依赖锁定文件，记录实际安装版本；
- 重新运行现有15项测试；
- 记录工作树提交、Python版本和测试结果；
- 确认模型权重和大张量不进入Git。

完成条件：

- 工作树中只有明确知道来源的改动；
- 15项测试继续通过；
- 模型权重不会被误提交。

#### A2：冻结TCP v1唯一规范

新建：

```text
docs/protocol/attention_tcp_v1.md
python/attention_protocol.py
tests/test_attention_protocol.py
vitis/attention_server/src/attention_protocol.h
```

实施内容：

- 将第4.4节逐字段写入协议文档；
- 用显式字节读写实现Python和C常量；
- 加入一组固定请求头和响应头的十六进制测试向量；
- 加入Q、K、V、Context的CRC测试向量；
- 测试错误Magic、Version、Command、Header长度、载荷长度、Token数、层号和CRC；
- 在Python测试中断言请求头40字节、响应头32字节。

完成条件：

- Python协议测试全部通过；
- C/Python字段偏移和常量逐项一致；
- 后续模块不再自行复制或重新定义协议常量。

#### A3：冻结Context布局

在`tests/test_llama_post_layer.py`中先建立哨兵数据：

```text
input[h,t,d] = 可唯一反推出 h、t、d 的值
```

验证：

```text
output[0,t,h*128+d] = input[h,t,d]
```

覆盖`valid_tokens=1`、短序列和128。

完成条件：任何后半层计算开始前，Context拼头规则已经由测试固定。

### 阶段B：纯PC参考链路

#### B1：复核PC前半层

主要文件：

```text
python/prepare_llama3_qkv.py
tests/test_prepare_llama3_qkv.py
```

复核内容：

- `residual_hidden`确实在input RMSNorm之前保存；
- Q/K/V为RoPE前数据；
- Q/K/V输出布局分别为`[32,128,128]`和`[8,128,128]`；
- BF16使用round-to-nearest-even并以小端保存；
- 第`valid_tokens`行以后全部为零；
- Manifest记录Token IDs、模型身份、计算精度、形状、CRC和SHA-256；
- 拒绝`L=0`、`L>128`、左补零和非零position offset。

如现有实现满足要求，只补测试和文档，不重写代码。

#### B2：实现Context解码和权重选择性加载

新建：

```text
python/llama_post_layer.py
tests/test_llama_post_layer.py
```

只加载第0层：

```text
self_attn.o_proj.weight
post_attention_layernorm.weight
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

接口建议：

```python
decode_context_bf16(payload, *, valid_tokens) -> torch.Tensor
load_post_weights(model_dir, layer_idx=0) -> PostWeights
run_post_attention(context, residual_hidden, weights, *, rms_eps) -> Layer0PostResult
```

`Layer0PostResult`保留：

```text
context_merged
o_proj
residual_1
post_norm
gate
up
swiglu
down_proj
layer_output
```

#### B3：实现PC后半层

固定顺序：

```text
Context
→ o_proj
→ + residual_hidden，得到residual_1
→ post-attention RMSNorm
→ silu(gate_proj(x)) * up_proj(x)
→ down_proj
→ + residual_1
```

单元测试覆盖：

- `L=1`、短输入、`L=128`；
- Context少2字节、多2字节；
- 非法`valid_tokens`；
- 权重形状错误；
- NaN/Inf；
- Residual来自RMSNorm前；
- `silu(gate) * up`顺序；
- 小型合成权重逐检查点参考。

同一CPU FP32输入下，对参考实现使用：

```text
rtol = 1e-5
atol = 1e-6
```

#### B4：实现两套Attention参考

新建：

```text
python/llama_attention_reference.py
tests/test_llama_attention_reference.py
```

两套参考分别为：

1. 标准PyTorch FP32 Attention，用于判断算法语义；
2. 镜像RTL各阶段BF16舍入、布局和Mask的位感知Golden，用于判断硬件实现。

两套参考不得混为同一个验收标准。

### 阶段C：PC协议客户端

#### C1：实现请求发送

新建或修改：

```text
python/fpga_attention_client.py
tests/test_fpga_attention_client.py
```

接口：

```python
send_qkv_request(sock, prepared_inputs, request_id) -> None
```

发送前验证：

- Q/K/V固定长度；
- `valid_tokens`范围；
- `layer_index=0`；
- 输入CRC与Manifest一致；
- 请求头字段完整。

使用`sendall()`发送头部、Q、K、V。

#### C2：实现响应接收

接口：

```python
recv_exact(sock, size) -> bytes
recv_context_response(sock, request_id) -> ContextResponse
```

接收顺序：

1. 精确接收32字节响应头；
2. 验证Magic、Version、Command、Header长度和request ID；
3. 错误状态直接返回错误，不读取伪Context；
4. 成功时精确接收1 MiB Context；
5. CRC通过后才允许解析BF16。

测试覆盖1字节分片、随机分片、合包、短读、EOF、超时、错误ID、错误长度、错误CRC和中途断流。

#### C3：本机模拟闭环

使用`socket.socketpair()`或本机模拟服务器验证：

- 固定Q/K/V请求共1.5 MiB逐字节一致；
- 固定Context响应1 MiB逐字节一致；
- 连续10次请求结果一致；
- 慢速接收和`recv_exact()`行为正确。

### 阶段D：PS正式Attention服务器

#### D1：建立正式Vitis/lwIP工程

建立：

```text
vitis/attention_server/
```

明确并记录：

- 匹配的BIT/XSA及SHA-256；
- `psu_cortexa53_0` standalone Domain；
- BSP和lwIP版本及配置；
- GEM3、PHY、MDIO、RGMII delay和网口复位配置；
- 静态IP：板卡`192.168.10.10`，PC`192.168.10.100/24`；
- 链接脚本、ELF和ELF Map；
- 从XSA到ELF的可复现构建命令。

链接脚本必须排除共享DDR保留区，并用ELF Map核对程序、堆栈、heap、lwIP缓冲和DMA没有侵入。

#### D2：实现Q/K/V接收

新建：

```text
qkv_rx.c/.h
```

要求：

- 支持任意`pbuf`分片和合包；
- 头部解析与载荷写入分阶段进行；
- 按Q、K、V固定区域写入DDR；
- 分别增量计算三个CRC；
- 所有载荷完整且CRC正确前禁止启动PL；
- 断流时丢弃未完成事务。

#### D3：抽取非阻塞PL控制

新建：

```text
fpt_attention_hw.c/.h
```

从`fpt_attention_board_test.c`提取并重新封装：

```c
fpt_hw_prepare_and_start(...)
fpt_hw_poll(...)
fpt_hw_enter_fault(...)
```

要求：

- Reset、Start和Poll不能在lwIP回调中长时间忙等；
- 每次`attention_server_poll()`只推进有限工作；
- 成功条件严格使用第4.3节；
- 超时和错误进入Fail-stop；
- 原板端测试继续保留并可独立运行。

#### D4：实现Context异步发送

新建：

```text
context_tx.c/.h
```

接口：

```c
context_tx_init(...)
context_tx_begin(...)
context_tx_pump(...)
context_tx_on_sent(...)
context_tx_done(...)
context_tx_abort(...)
```

规则：

- `attention_server`唯一拥有`tcp_pcb`和所有lwIP回调；
- `context_tx`不保存、关闭或中止PCB；
- `begin()`只处理成功Context，不生成通用错误响应；
- Invalidate完整Context后计算CRC并构造响应头；
- 先发送响应头，再发送Context；
- 每次`pump()`最多提交一个块；
- 块长不超过`min(remaining, 16 KiB, tcp_sndbuf())`；
- 使用`TCP_WRITE_FLAG_COPY`；
- `tcp_write()==ERR_MEM`时不推进偏移，等待后续回调或轮询；
- 分别记录已提交字节数和已ACK字节数；
- 只有头部和Context全部收到ACK后才算完成；
- `abort()`幂等且不访问PCB。

#### D5：实现中央状态机

新建：

```text
attention_server.c/.h
main.c
```

状态机：

```text
WAIT_HEADER
→ RX_Q → RX_K → RX_V → VALIDATE
→ PREPARE_HW → RUN_HW
→ TX_HEADER → TX_CONTEXT
→ WAIT_HEADER

不可安全恢复的硬件错误
→ FAULT
```

要求：

- `attention_server`注册`tcp_recv`、`tcp_sent`、`tcp_err`；
- lwIP回调只更新状态，不阻塞等待PL；
- 主循环处理GEM/lwIP输入和计时器后调用`attention_server_poll()`；
- `tcp_err`发生时PCB已释放，禁止再次访问；
- 新连接不能绕过`FAULT`；
- Context完全ACK前禁止覆盖共享DDR。

### 阶段E：目标模型、RoPE与硬件资产

#### E1：验收队友提供的模型包

使用`python/prepare_llama31_model.py`检查：

- repo ID和固定revision；
- 配置维度为32层、hidden 4096、32Q/8KV、head_dim 128、intermediate 14336；
- 第0层所需tensor到shard映射完整；
- 每个文件大小和SHA-256；
- `rope_theta`及`rope_scaling`或`rope_parameters`；
- 无QKV和MLP bias；
- Tokenizer和配置来自同一revision。

只有生成状态为`complete`的模型清单后，才能进入正式RoPE资产替换。

#### E2：生成Llama 3.1 RoPE LUT

新建：

```text
python/generate_llama31_rope_lut.py
tests/test_llama31_rope_lut.py
```

要求：

- 只读取交付包中的原始`config.json`；
- 同时兼容`rope_scaling`和`rope_parameters`字段；
- 验证类型为`llama3`；
- 验证theta、factor、low/high factor和original length；
- 生成position `0..127`和64个频率的sin/cos；
- 使用BF16 round-to-nearest-even；
- 输出各8192项的hex文件；
- 记录配置摘要和ROM SHA-256；
- 共16,384项与独立参考逐字一致。

生成器第一步只输出到临时目录，不直接覆盖现有`mem/*.hex`。

#### E3：保存旧硬件基线

在替换ROM前运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tests\run_v313_qk4_system_checks.ps1
.\01_check_rtl.bat
.\02_build_bitstream.bat
.\03_build_vitis.bat
```

随后完成：

- 一次预热；
- 十次正式实板运行；
- 保存BIT/XSA/ELF、UART日志和SHA-256；
- 缺少工具或板卡时明确记录`NOT RUN`。

#### E4：替换ROM并生成同源Golden

- 覆盖`mem/sin_bf16.hex`和`mem/cos_bf16.hex`；
- 修正并验证Golden工具的BF16 RNE；
- 生成与新ROM同源的位感知Context Golden；
- 保存模型配置、ROM和Golden哈希；
- 不改变RoPE旋转的数据通路RTL，除非测试证明现有实现与目标定义不符。

#### E5：重新构建和实板验收

依次完成：

```text
RoPE单元测试
→ Host/Python测试
→ RTL回归
→ RTL展开检查
→ Vivado综合与实现
→ Timing/DRC
→ BIT/XSA
→ 板端测试ELF和服务器ELF
→ 预热与10次实板测试
```

## 7. 分层联调

### J1：PC后半层

- 同一Context和Residual分别送入自定义实现和参考实现；
- 逐项比较`o_proj`、第一次Residual、RMSNorm、gate/up、SwiGLU、down_proj和最终Residual；
- 满足`rtol=1e-5, atol=1e-6`。

### J2：TCP基线

- 短消息和1 MiB Echo各重复10次；
- 每次逐字节一致；
- 覆盖慢速接收、任意分片和中途断开。

### J3：确定性Context回传

- PS在Context DDR填入可预测模式；
- 经过正式响应头和`context_tx`返回PC；
- 验证长度、request ID、状态、CRC和布局；
- 错误后行为符合Fail-stop或安全关闭规则。

### J4：当前ROM对应Golden闭环

- 使用与当前ROM匹配的仓库Q/K/V；
- 运行PL并通过TCP返回Context；
- 对比对应位感知Golden；
- 核对完整Context CRC和FPGA状态。

### J5：目标Llama 3.1 Attention闭环

- 使用目标模型第0层生成真实Q/K/V和Residual；
- 先核对PC文件CRC与PS DDR落地CRC；
- FPGA Context对比同源位感知Golden；
- 前`valid_tokens`行对比标准PyTorch Attention。

### J6：完整第0层闭环

新建统一入口：

```text
python/run_layer0_hybrid.py
```

执行：

```text
文本或Token IDs
→ Q/K/V + residual_hidden
→ TCP请求FPGA
→ 接收Context
→ PC后半层
→ 保存检查点与报告
```

在`out/<request_id>/`保存：

- 模型、配置、Tokenizer和权重清单哈希；
- Token IDs和`valid_tokens`；
- ROM、Golden、BIT/XSA/ELF哈希；
- 请求头、响应头和CRC；
- FPGA状态；
- 每阶段耗时；
- 数值指标与NaN/Inf检查结果。

### J7：错误注入

覆盖：

- 错误Magic、Version、Command和Header长度；
- 错误Q/K/V长度；
- 非法Token数和层号；
- 输入CRC错误；
- BUSY和流水请求；
- 请求中途断流；
- Context发送中途断流；
- Reset超时、运行超时和FPGA错误；
- 进入`FAULT`后重连仍不能启动新请求；
- 经过已验证的系统复位后才能恢复。

## 8. 验收标准

### 8.1 配置与RoPE

- 模型配置与硬件维度完全匹配；
- sin/cos各8192项，共16,384项逐字匹配独立参考；
- 配置、ROM、Golden、BIT/XSA/ELF哈希能够闭合追踪。

### 8.2 Attention

对全部524,288个Context BF16元素：

```text
FPGA与位感知Golden：
abs_error <= 1e-4 或 BF16 ULP <= 1
combined_failures = 0
```

前`valid_tokens`行还需与标准PyTorch Attention比较。若失败，先定位差异来源，不自动放宽容差。

### 8.3 PC后半层

同一CPU FP32输入与权重下：

```text
rtol = 1e-5
atol = 1e-6
```

逐检查点满足容差且无NaN/Inf。

### 8.4 系统

- 请求与响应的ID、长度、状态和CRC正确；
- FPGA成功状态完整且无错误位；
- 同一请求连续10次返回完全相同的BF16 Context；
- 错误注入产生确定行为；
- `FAULT`不会被断线、重连或普通软复位错误清除；
- 完整混合第0层记录最大/平均绝对误差、RMSE和余弦相似度。

### 8.5 构建

- RTL展开检查通过；
- Vivado综合和实现完成；
- WNS不小于0；
- 无未约束时序路径；
- DRC错误为0，严重警告逐项说明；
- 板端测试ELF和服务器ELF使用匹配XSA；
- 保留实际UART日志和所有工件SHA-256。

## 9. 提交策略

每个工作包独立提交，避免把协议、PC算法、PS网络和硬件资产混在同一个大提交中。建议提交顺序：

```text
1. 环境与忽略规则
2. TCP v1协议文档、Python/C常量和测试
3. Context布局与PC后半层
4. PC Attention参考
5. PC客户端和模拟网络测试
6. qkv_rx
7. fpt_attention_hw
8. context_tx
9. attention_server和Vitis工程
10. 目标模型清单与RoPE生成器
11. ROM、Golden和硬件构建资产
12. 分层联调入口和报告
```

模型权重、完整Q/K/V/Context、BIT、XSA、ELF和大体积输出不提交Git；仓库只保存代码、小型测试向量、Manifest、哈希和报告。

## 10. 立即开始时的第一项工作

正式执行时先完成A1和A2，不先写PC后半层，也不先修改RoPE ROM：

1. 整理并提交`.gitignore`和依赖版本；
2. 重新运行现有15项测试；
3. 新建`docs/protocol/attention_tcp_v1.md`；
4. 新建Python协议打包/解包和固定字节测试；
5. 新建C端协议常量头文件；
6. 确认Python/C测试向量逐字一致。

完成协议冻结后，再进入Context布局和PC后半层。这样后续PC客户端、PS接收和Context回传都使用同一份协议，不需要返工。

## 11. 后续路线

第0层全部门禁通过后，再单独规划：

1. PC前后半层参数化到任意`layer_idx`并实现32层循环；
2. 增加最终RMSNorm、LM Head和贪心输出；
3. 重新设计支持KV Cache、`q_len=1`、动态KV长度和非零position offset的PL和TCP v2。

这些内容不在当前计划中提前实现。
