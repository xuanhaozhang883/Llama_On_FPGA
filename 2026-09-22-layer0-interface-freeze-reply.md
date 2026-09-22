# 第 0 层 PC–FPGA–PC：接口冻结信息回复

核对日期：2026-09-22

> 历史核对快照：本文记录代码整理提交之前的状态。后续目标、Fail-stop 和责任确认以 `docs/layer0_handoff_contract.md` 为准，代码基线以本次推送后的完整提交 SHA 为准。

我重新核查了本机工作区，以下按已确认事实和待落实事项回复。前两组已有实测数据；第三组还有工程和硬件恢复机制待落实，暂时不能写成“已经冻结”。

## 1. 当前代码版本

本机执行 `git rev-parse HEAD` 的结果是：

```text
bad215b8b621516564933624b6355a642b2218d6
```

`git status --short` 的结果是：

```text
 M README_CN.md
 M python/prepare_llama3_qkv.py
 M tests/test_prepare_llama3_qkv.py
?? python/prepare_llama3_embedding.py
?? python/quantize_llama3_embedding.py
?? python/safetensors_stream.py
?? python/tokenize_llama3.py
?? python/verify_llama3_embedding.py
?? scripts/run_embedding.ps1
```

以上为编写本回复文件前的核对快照，不包含本回复文件本身。

**`prepare_llama3_embedding.py` 尚未提交，目前是本地未跟踪文件。** QKV 脚本也有未提交修改。因此，即使我们 HEAD 相同，当前工作区内容也可能不同。之前“双方当前仓库已经一致”的表述需要修正。

正式共同基线应以这些改动整理提交后的具体 SHA 为准，不能直接以目前的本地工作区作为冻结版本。

## 2. 当前实际模型与文件哈希

当前使用的本地目录是 `D:\Llama_weight`，`config.json` 相关内容如下：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 8192,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "rope_scaling": null,
  "torch_dtype": "bfloat16",
  "vocab_size": 128256
}
```

**该文件不存在 `rope_parameters` 字段。**

本次实际计算的 SHA-256：

```text
config.json
2430cee764b6530ff8673cf9ba8561e1d5a33152d503cd0de909ff5718261441

model.safetensors.index.json
146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b
```

关于来源，目前能核实到的证据是：项目下载说明（`README_CN.md` 第 119 行）指向 Hugging Face 的 `meta-llama/Meta-Llama-3-8B`；本地 config 和 tokenizer 的下载元数据记录了 revision：

```text
8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
```

这个 revision 不能直接当作全部权重文件已经核验的版本。当前目录只有 `model-00001-of-00004.safetensors` 一个正式权重分片；索引显示 Embedding 和第 0 层所需权重均位于该分片，但权重分片的下载来源及 revision 还没有完整的可核验记录。

**因此，目前不能确认本地模型已经统一为 `Llama-3.1-8B-Instruct`。上述哈希只是当前本地文件的哈希。目标模型及其 RoPE 配置共同确认前，暂不替换 RoPE ROM。**

## 3. 正式 Attention 服务器、Reset 和 DDR 预留

### 3.1 工程配置与构建方式

正式服务器尚未创建，当前已知配置和拟定方案如下：

| 项目 | 当前事实／拟定方案 |
|---|---|
| XSA | 预期使用 `D:\学习\Llama_On_FPGA\export\fpt_attention_board_v313_qk4_flashattention.xsa`，但当前文件不存在，需要从确定的源码版本重新生成并记录哈希。 |
| A53 | 拟使用 `psu_cortexa53_0`、`standalone`。 |
| Domain | 现有测试脚本使用 `standalone_domain`，Echo 文档使用 `standalone_psu_cortexa53_0`；正式工程需选定并以实际生成结果为准，目前未统一。 |
| BSP | 拟使用支持 lwIP 的 standalone BSP；正式 BSP 尚未生成，版本和参数待提供。 |
| GEM | 当前 BD 已配置 GEM3，RGMII 使用 MIO 64–75，MDIO 使用 MIO 76–77。 |
| PHY | 型号、MDIO 地址、复位方式及 RGMII delay 配置尚待板卡资料和实测核实。 |
| 工程路径 | 建议源码放在 `D:\学习\Llama_On_FPGA\vitis\attention_server\src`，该目录尚未创建。 |

构建方面，现有 `02_build_bitstream.bat` 是 BIT/XSA 构建入口；`03_build_vitis.bat` 当前只构建 `fpt_attention_test`，不是网络服务器。正式服务器计划基于官方 lwIP Echo 模板建立，生成平台/BSP 后集成各模块，再补充可复现的服务器构建入口和 ELF 路径。

### 3.2 Reset 后 DDR 安全状态

**Reset 后 DDR 是否安全，目前存在明确缺口。** 当前驱动以：

```text
READY = 1
BUSY = 0
DONE = 0
ERROR = 0
```

判断复位完成。但 RTL 中的 READY（`rtl/board/attention_board_top.sv` 第 99 行附近）实际上只是 `!engine_busy`；软复位会直接清空状态机，没有提供 AXI 未完成读写事务已经排空的确认。因此，这组状态目前不能作为“PL 已停止访问 DDR、缓冲区可以立即复用”的保证。

建议正式冻结的恢复语义是：先停止发起新事务，按 AXI 协议完成在途事务，再提供 `DDR_QUIESCENT/RESET_DONE` 或等价的可验证状态。**该机制实现并验证前，运行异常或超时后进入故障状态，不自动启动下一请求，也不覆盖共享 DDR；恢复使用经过验证的系统复位流程。**

### 3.3 链接脚本中的 DDR 预留

链接脚本方面，拟从普通可分配 DDR 中扣除：

```text
[0x10000000, 0x10280000)
```

第一版优先将程序、静态数据、heap、stack、lwIP 内存及网卡 DMA 缓冲限制在该区间以下的已确认可用 DDR 中，并检查驱动是否另外使用固定地址。根据实际 BSP 的堆栈边界增加链接断言，最终提供 `lscript.ld` 和 ELF map 核对。

不能仅添加一个 `NOLOAD` 段就认为完成隔离，还必须限制运行时分配器的范围；布局可以通过 GNU ld 的 [MEMORY](https://sourceware.org/binutils/docs/ld/MEMORY.html) 和 [ASSERT](https://sourceware.org/binutils/docs/ld/Miscellaneous-Commands.html) 约束落实。
