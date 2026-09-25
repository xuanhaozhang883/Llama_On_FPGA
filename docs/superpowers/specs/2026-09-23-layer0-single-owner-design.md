# Llama 3.1 8B 第 0 层 PC–FPGA–PC 单人实施设计

> **历史文档，已被取代：** 本设计保留用于追溯旧Llama 3.1目标，不再作为当前事实或实现依据。当前目标以`docs/superpowers/specs/2026-09-24-llama3-target-migration-design.md`为准。

> 日期：2026-09-23  
> 状态：设计基线，尚未开始功能实现  
> 目标：在保留模块边界的前提下，由一人负责第 0 层 PC–FPGA–PC 闭环；队友只提供目标模型文件。

## 1. 目标与范围

第一里程碑验证 Llama 3.1 8B Instruct 的第 0 个 Transformer 层：

```text
PC（CPU FP32）
Tokenizer → Embedding → input RMSNorm → Q/K/V Projection
                         │
                         │ RoPE 前 Q/K/V，BF16/TCP
                         ▼
PS（A53 裸机 + lwIP）
接收 Q/K/V → 写入 DDR → Flush → 启动并轮询 PL
                         │
                         ▼
PL
Llama 3.1 scaled RoPE → GQA Attention → Context
                         │
                         │ Context，BF16/TCP
                         ▼
PC（CPU FP32）
Context 拼头 → o_proj → Residual → RMSNorm
→ SwiGLU MLP → Residual → 第 0 层输出
```

本阶段不实现：

- 32 层循环；
- KV Cache 和逐 Token Decode；
- 动态序列长度的硬件接口；
- 最终 RMSNorm、LM Head、采样和文本生成；
- GPU 性能优化；
- 多客户端或多请求并发。

## 2. 责任边界

### 2.1 唯一外部输入

队友只负责提供目标模型文件。交付包至少包含：

- 官方 `meta-llama/Llama-3.1-8B-Instruct` 的 `config.json`；
- Tokenizer 相关文件；
- `model.safetensors.index.json`；
- Embedding 和完整第 0 层所需的全部权重分片；
- 模型仓库、固定 revision、文件大小和 SHA-256。

目标 revision 暂按现有交付约定记录为：

```text
0e9e39f249a16976918f6564b8830bc894c89659
```

收到后必须自行校验，不能因为文件来自队友就直接认定正确。当前已有的 `model-00001-of-00004.safetensors` 属于旧 Llama 3 模型，只能用于冒烟测试，不能用于正式 Llama 3.1 RoPE、Golden 或最终验收。

### 2.2 项目内全部责任

除模型文件外，以下工作全部由项目负责人完成：

- 模型资产检查及固定测试输入；
- PC 前半层和 `residual_hidden`；
- PC 后半层；
- TCP v1 协议及 PC 客户端；
- PS 端 Q/K/V 接收、DDR 管理和 Cache 操作；
- PL Reset、Start、Poll 和状态判断；
- PS 端 Context 异步发送；
- `attention_server` 中央状态机；
- Llama 3.1 RoPE LUT、两套 Golden 和硬件资产更新；
- Vitis/lwIP 正式服务器工程；
- 分层联调、实板验证、报告和最终验收。

责任集中不意味着把代码写进一个文件。各模块仍通过固定接口连接，以便独立测试和定位问题。

## 3. 当前基线

后续工作以提交 `e20d221b722be6e0b88295279041607cc7bbcfc2` 为代码起点。

已经具备：

- `python/prepare_llama3_qkv.py`：能够生成 RoPE 前 Q/K/V、RMSNorm 前 `residual_hidden` 和 Manifest；
- `python/prepare_llama31_model.py`：能够固定 revision、解析索引并检查模型交付包；
- 相关 15 项 Python 单元测试已在 Python 3.12.10 环境中通过；
- 固定 `32Q/8KV/S128/D128` 的 PL Attention 数据通路；
- `vitis/src/fpt_attention_board_test.c` 中已有 DDR、Cache、Reset、Start、Poll 和 Context 检查参考逻辑；
- `vitis/eth_echo` 中已有 lwIP Echo 示例；
- RTL、BIT/XSA 和现有板端测试的构建入口。

尚未具备或尚未正式验收：

- 正式 `vitis/attention_server` 工程；
- TCP v1 的 Python/C 公共协议实现；
- PC Context 接收和 PC 后半层；
- `qkv_rx`、`fpt_attention_hw`、`context_tx` 和中央状态机；
- 与目标 Llama 3.1 配置同源的 RoPE ROM 和 Golden；
- 正式 Attention 服务器的 XSA/BSP/ELF/Map 与实板闭环证据。

“已有代码”只表示可以继承和改造，不等于已经通过目标模型或端到端验收。

## 4. 固定系统接口

### 4.1 张量接口

| 张量 | 类型与布局 | 字节数 |
|---|---|---:|
| Q | BF16 小端 `[32,128,128]` | 1,048,576 |
| K | BF16 小端 `[8,128,128]` | 262,144 |
| V | BF16 小端 `[8,128,128]` | 262,144 |
| Context | BF16 小端 `[32,128,128]` | 1,048,576 |
| `residual_hidden` | CPU FP32 `[1,L,4096]` | 仅在 PC 内存中使用 |

固定语义：

- PC 输出 RoPE 前的 Q/K/V；RoPE 只在 PL 执行一次；
- `residual_hidden` 是 input RMSNorm 前的 Embedding 输出；
- 固定硬件长度为 128，`valid_tokens=L`，第 `L` 行及以后右补零；
- Context 是 `o_proj` 之前的每头 Attention 输出；
- Context 进入 `o_proj` 前执行 `[head,token,dim] → [token,head,dim] → [1,L,4096]`。

### 4.2 DDR 与 Cache

| 区域 | 地址 | 长度 |
|---|---:|---:|
| Q | `0x10000000` | `0x100000` |
| K | `0x10100000` | `0x040000` |
| V | `0x10140000` | `0x040000` |
| Context | `0x10180000` | `0x100000` |

整个 `[0x10000000,0x10280000)` 必须从程序、堆栈、heap、lwIP 缓冲和 DMA 可分配范围中排除。

Cache 所有权固定为：

- PS 写完 Q/K/V 后、PL 读取之前执行 Flush；
- Context 启动前的清零和必要 Flush 由硬件控制模块负责；
- PL 完成写入后、PS 读取 Context 之前执行 Invalidate；
- 同一块缓冲区在所属事务完成前不得复用或覆盖。

### 4.3 TCP v1

固定使用端口 5001、同一 TCP 连接、单客户端、单请求在途、全部多字节整数小端编码。CRC 为：

```python
zlib.crc32(payload) & 0xffffffff
```

请求头固定 40 字节，随后为 `Q || K || V`；成功响应头固定 32 字节，随后为完整 1 MiB Context。协议实现必须包含固定十六进制样例和跨语言测试，禁止直接发送可能含 C 填充字节的结构体。

### 4.4 模块边界

```text
PC
├─ prepare_llama3_qkv.py       前半层
├─ llama_post_layer.py         Context 解码与后半层
├─ attention_protocol.py       协议打包、解包和 CRC
├─ fpga_attention_client.py    TCP 请求与响应
└─ run_layer0_hybrid.py        最终统一入口

PS
├─ attention_protocol.h        C 端协议常量与小端辅助
├─ qkv_rx.c/.h                 请求解析、DDR 写入、输入 CRC
├─ fpt_attention_hw.c/.h       Cache、Reset、Start、Poll
├─ context_tx.c/.h             Context Invalidate、CRC、异步发送
└─ attention_server.c/.h       lwIP 回调和中央状态机
```

`attention_server` 唯一拥有 `tcp_pcb`、连接生命周期和 lwIP 回调。`context_tx` 只负责成功响应的数据发送，不关闭连接、不注册回调，也不拥有 PCB。

## 5. 错误处理

第一版采用 Fail-stop：

- 可以证明 PL 尚未启动的协议错误，可以返回错误头；只有完整消费当前帧且边界明确时才允许复用连接，否则关闭连接；
- Start 已发出，或无法排除已发出后发生超时/硬件错误，服务器进入 `FAULT`；
- `FAULT` 下禁止接收新请求，禁止覆盖共享 DDR，断线重连也不能清除故障；
- 只有经过验证的系统复位，并确认 PL/AXI 不再访问共享 DDR 后，才能恢复服务；
- `READY=1`、`BUSY=0` 或软复位完成不能单独作为 AXI 已静止的证明。

这取代旧计划中“任何错误后下一条请求都能自动恢复”的表述。

## 6. 实施策略

采用“模块化分阶段推进，并先建立纯 PC 参考”的路线。

### 阶段 0：基线与接口冻结

- 固定代码起点、Python 环境和依赖；
- 将本设计中的张量、DDR、协议、Cache、状态和错误语义写入唯一协议规范；
- 建立协议固定字节测试和 Context 布局哨兵测试；
- 保留旧 ROM，不在模型包到达前替换硬件资产。

完成条件：任何模块都可以只依据公开接口独立实现和测试。

### 阶段 1：纯 PC 数值链路

- 复核现有 PC 前半层；
- 实现 Context 解码和 PC 后半层；
- 实现标准 PyTorch Attention 与镜像 RTL 舍入过程的位感知 Golden；
- 使用合成小模型和旧模型分片做功能测试；
- 收到目标模型包后执行正式配置、权重和数值验证。

完成条件：不用板卡，也能验证前半层、Attention参考和后半层的每个检查点。

### 阶段 2：协议与 PC 客户端

- 实现 Python/C 公共协议常量；
- 实现 `send_qkv_request()`、`recv_exact()` 和 `recv_context_response()`；
- 使用本机模拟服务器覆盖任意分片、短读、CRC、错误头和断连；
- 传输层始终校验完整固定大小载荷。

完成条件：在无板卡环境下，协议和 1 MiB 数据往返可重复通过。

### 阶段 3：PS 正式服务器

- 从现有板端测试抽取非阻塞硬件控制接口；
- 实现 `qkv_rx`、`fpt_attention_hw`、`context_tx`；
- 实现中央状态机和 Fail-stop；
- 建立正式 Vitis/lwIP 工程、链接脚本和可复现构建入口；
- 保留原板端测试程序作为独立硬件回归。

完成条件：PS 能接收固定测试数据、驱动 PL，并把确定性 Context 模式可靠返回 PC。

### 阶段 4：目标 RoPE 与硬件资产

此阶段依赖队友交付并通过校验的目标模型包。

- 从原始 `config.json` 解析 Llama 3.1 scaled-RoPE；
- 生成 BF16 RNE sin/cos LUT；
- 与独立参考逐项比较；
- 生成与 ROM 同源的位感知 Golden；
- 保存旧基线证据后再替换 ROM；
- 重新完成 RTL 回归、Vivado 实现、Timing/DRC、BIT/XSA/Vitis 和实板测试。

完成条件：模型配置、LUT、Golden、BIT/XSA/ELF 的来源和哈希可以闭合追踪。

### 阶段 5：分层联调

按以下顺序进行，前一级失败时停在本级排查：

1. PC 后半层独立验证；
2. TCP 短消息和 1 MiB 传输验证；
3. PS 确定性 Context 模式返回；
4. 当前 ROM 对应的旧 Golden 闭环；
5. 目标 Llama 3.1 Q/K/V 与 Attention 闭环；
6. 完整第 0 层闭环；
7. 错误注入与 Fail-stop 验证。

### 阶段 6：最终验收

最终报告必须分别记录：

- 模型、配置、Tokenizer、权重和测试输入身份；
- Q/K/V、Residual、ROM、Golden、BIT/XSA/ELF 哈希；
- PC 前半层、Attention、PC 后半层的数值指标；
- QKV 上传、PL 计算、Context 下载和 PC 后半层耗时；
- FPGA 状态、十次重复一致性和错误注入结果；
- 未运行项目必须明确标记 `NOT RUN`，不能写成通过。

## 7. 执行顺序与阻塞关系

模型包未到达时可以推进：

- 协议冻结和协议单元测试；
- Context 布局和 PC 后半层合成测试；
- PC 客户端及模拟服务器；
- PS 服务器模块和确定性 Context 发送；
- 旧 ROM/旧 Golden 的回归基线。

模型包到达后才能正式推进：

- 目标模型身份验收；
- Llama 3.1 RoPE LUT 与 ROM 替换；
- 目标模型 Q/K/V、Residual 和 Attention Golden；
- 正式第 0 层数值验收。

因此，模型交付是正式验收的阻塞项，但不是开始开发的阻塞项。

## 8. 完成定义

只有同时满足以下条件，才能称为“完成第 0 层 PC–FPGA–PC 闭环”：

- 使用经过校验的目标 Llama 3.1 8B Instruct 第 0 层模型资产；
- PC 发送的 Q/K/V 确认是 RoPE 前数据；
- FPGA 返回完整、布局正确的 `o_proj` 前 Context；
- PC 后半层逐检查点与参考实现满足既定容差；
- TCP 的长度、request ID、状态和 CRC 全部正确；
- FPGA 完成状态完整且无错误；
- 同一请求连续 10 次返回完全相同的 BF16 Context；
- Fail-stop 和系统复位恢复行为通过实板验证；
- 构建工件、配置、ROM、Golden 和模型身份能够通过哈希追踪。

通过 15 项现有单元测试、运行 TCP Echo 或单独跑通板端 Golden，都不能单独代表上述闭环已经完成。
