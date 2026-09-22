# 第 0 层前半层交付与责任确认

日期：2026-09-22

状态：前半层交付基线与责任约定；待双方拉齐代码后建立最终接口冻结提交。

本文中的“我方”指本仓库的前半层与正式 Attention 服务器负责人，“队友”指后半层与 Context 发送模块的协作方。本文确认目标、恢复策略和交付责任，不表示正式服务器、接口冻结或实板验收已经完成。

## 1. 模型目标与材料交付

确认目标保持为 **`meta-llama/Llama-3.1-8B-Instruct`**。

我方负责从官方仓库取得并交付同一固定 revision 的以下材料：

- 原始 `config.json`，完整保留实际 RoPE 配置。
- Tokenizer 文件，包括 `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`，以及该 revision 使用的其他相关文件。
- `model.safetensors.index.json`。
- Embedding 和完整 decoder 第 0 层所需的全部权重分片；包含双方前、后半层需要的张量，不仅限于 Q/K/V 投影。
- 逐文件 SHA-256、文件大小、官方来源 URL、完整 revision，以及张量到分片的映射。

本次从官方仓库解析到的 revision 是：

```text
0e9e39f249a16976918f6564b8830bc894c89659
```

该 revision 对应的官方来源：

<https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659>

**当前下载状态：本机账户请求该仓库的 `config.json` 返回 HTTP 403，尚未取得完成授权的模型文件包。** 上述 revision 是来源版本信息，不能作为实际文件已经下载或校验通过的证明。需要具备该官方受限仓库访问权限的账户完成获取；材料交付责任仍由我方承担。

仓库中的 `python/prepare_llama31_model.py` 用于固定 revision、选择 Embedding 与完整第 0 层所需分片、下载及核对文件。只有生成状态为 `complete` 的 `model-bundle-manifest.json`，才能作为完成的模型包交付；`bundle-plan.json` 只是下载计划。

双方对照模型包清单及文件 SHA-256，确认实际 `rope_theta`、`rope_scaling` / `rope_parameters` 后，再安排与该模型对应的 RoPE LUT/ROM 和同源 Golden 验证。**在此之前不替换 RoPE ROM，不修改旧模型的 config 来冒充目标模型。**

正式联调使用原始权重导出的 Embedding，禁用 INT8 Embedding 演示产物。前半层联调计算固定为 CPU FP32，PyTorch CPU 线程数为 1；Manifest 同时记录软件版本和实际计算精度。跨机器数值对齐仍以双方核对后的产物与误差验收标准为准。

前半层交付数据包括固定 Token IDs、`valid_tokens`、Q/K/V BF16 文件以及 `residual_hidden_fp32.npy`。Residual 是 input RMSNorm 之前的 Embedding，形状为 `[1,L,4096]`；Q/K/V 按固定 128 行的 `[head][token][dim]` 布局导出，第 `L` 行及以后补零。同份 Manifest 关联输入、模型身份与所有输出文件哈希。

## 2. 第一版采用 Fail-stop 恢复策略

确认第一版采用以下恢复语义：

| 发生阶段 | 处理与恢复条件 |
| --- | --- |
| 可以证明 PL 尚未启动的协议错误 | 只有当前请求完整帧已经消费、帧边界明确且不会把残留字节当作新请求时，才允许返回错误并复用连接。否则关闭连接，由新的连接重新开始。 |
| PL 的 Start 脉冲已经发出，或无法排除已经发出后发生超时、硬件错误 | 进入 `FAULT`，不接受下一请求，不写入或覆盖共享 Q/K/V、Context DDR 区域。 |
| `FAULT` 状态下断线、重连或客户端重试 | 保持 `FAULT`，不得借此清除故障或启动新请求。 |
| 故障后恢复 | 只有经过验证的系统复位流程完成，并确认 PL/AXI 对共享 DDR 的访问已停止后，才能重新初始化服务器、缓冲区并恢复接收请求。 |

进入 `FAULT` 后，可以发送不读取、不修改 Context 的错误响应元数据，或关闭连接；不能为了构造错误响应而读取可能仍在变化的 Context。

当前 RTL 的 `READY` 表示 `!engine_busy`，现有软复位会清空内部状态。因此 `READY=1`、`BUSY=0` 或软复位完成状态，**不能单独证明 AXI 在途事务已经排空，也不能证明 DDR 可以安全复用**。

最终恢复验收必须给出实际使用的系统复位步骤、覆盖 PS/PL/AXI 的范围，以及能够证明共享 DDR 不再被旧事务访问的检查或验证记录。该流程目前尚未验收；在验收之前，软件不得实现从 `FAULT` 自动恢复下一请求的路径。

## 3. 正式 Vitis/lwIP Attention 服务器责任

确认正式服务器工程由我方负责，包括：

- 对应源码版本的 BIT/XSA、文件 SHA-256 和硬件配置说明。
- A53 Domain、standalone BSP、lwIP 配置和版本。
- GEM/PHY、MIO、MDIO、网口复位及链路配置。
- 正式服务器源码、链接脚本、ELF、ELF Map。
- 从源码和 XSA 开始可复现的构建命令、工具版本及产物位置。
- 共享 DDR 布局隔离与 Fail-stop 行为的实现、检查和验收记录。

现有事实与待完成项如下：

| 项目 | 当前已知配置或交付计划 |
| --- | --- |
| 硬件目标 | `xczu15eg-ffvb1156-2-i`；项目配置指定 Vivado 2025.2。 |
| 目标 XSA | `export/fpt_attention_board_v313_qk4_flashattention.xsa`；需要按确定的源码版本实际生成、记录哈希后交付。文件名不表示产物已经存在或已经验收。 |
| A53 | 拟使用 `psu_cortexa53_0`、`standalone`。 |
| Domain | 现有测试脚本使用 `standalone_domain`，Echo 文档使用 `standalone_psu_cortexa53_0`；正式工程需生成并选定实际名称。 |
| BSP/lwIP | 正式 BSP 尚未生成；具体版本、库参数和生成配置待交付。 |
| GEM | 现有 BD 使用 GEM3；RGMII 为 MIO 64–75，MDIO 为 MIO 76–77。驱动实例宏以正式 XSA 生成的 BSP 为准，不按 GEM 名称猜测实例编号。 |
| PHY | 型号、MDIO 地址、复位方式、RGMII delay 和链路协商配置待板卡资料及实测核实。 |
| 正式源码路径 | 计划使用 `vitis/attention_server/src`；当前尚无完成的正式 Attention 服务器工程。 |
| 现有构建入口 | `02_build_bitstream.bat` 用于 BIT/XSA；`03_build_vitis.bat` 当前构建 `fpt_attention_test`，不能当作正式网络服务器构建命令。 |

最终交付时，我方提供实际验证过的正式服务器构建命令，不把“基于 lwIP Echo 创建工程”的说明当作可复现构建已经完成。

## 4. 链接布局与共享 DDR 隔离

确认普通程序与运行时分配布局必须排除整个半开区间：

```text
[0x10000000, 0x10280000)
```

该区间用于 Q/K/V/Context，共计 `0x280000` 字节。正式链接脚本应从普通可分配 DDR 的 `MEMORY` 区域中扣除这段地址，并将程序段、静态数据、heap、stack、lwIP 缓冲和网卡 DMA 缓冲放在经确认的其他可用 DDR 中。

不能只建立一个 `NOLOAD` 段就宣称隔离完成；还需要核对 BSP/运行时分配器的实际边界、驱动固定地址与 DMA 区域，确保任何对象都不跨入上述区间。最终提交链接脚本、适用的链接断言及 ELF Map，并给出各段和分配区域的地址核对结果。

## 5. 拉齐代码并冻结接口

- [ ] 我方整理并提交前半层代码，推送后提供分支名与完整 Git 提交 SHA；队友拉取并核对相同基线。
- [ ] 双方确认本文的模型材料责任、Fail-stop 语义、服务器工程责任和 DDR 排除区间。
- [ ] 确认 Token IDs、Q/K/V、Residual 及 Manifest 的形状、字节序、布局、有效长度和计算精度约定。
- [ ] 双方共同确认具体协议字段表、错误码和模块接口头文件，形成可以独立实现的定义；本文不预设尚未确认的接口字段。
- [ ] 拉齐代码与接口定义后，双方建立并共同记录最终接口冻结提交 SHA。

接口冻结确定双方独立实现所需的定义和责任，不以所有服务器实现、最终构建和实板验收已经完成为前提。以下是后续正式交付要求，不能误写为本次已完成。

## 6. 正式交付与实板验收清单

- [ ] 完成目标模型访问授权和下载，交付同一 revision 的完整第 0 层模型包、逐文件 SHA-256 与完成清单；双方核对一致后才推进新 RoPE ROM。
- [ ] 提供正确目标模型的同源 Token IDs、Q/K/V、Residual 和 Manifest。
- [ ] 生成并交付正式 XSA、A53 Domain、BSP/lwIP、GEM/PHY 配置、服务器源码、链接脚本、ELF 和 ELF Map。
- [ ] 验证普通内存、heap/stack、lwIP 和 DMA 区域均不侵入保留 DDR。
- [ ] 提供可复现构建命令与工具版本，完成对应产物的构建检查和实板验收。
- [ ] 验证协议错误的帧边界恢复、PL 启动后故障进入 `FAULT`、重连不能清故障，以及经过验证的系统复位恢复流程。

以上复选项表示待双方核对的要求，不表示已经实现、构建通过或实板验证通过。前半层代码交付、最终接口冻结、完整服务器实板验收是不同的里程碑。
