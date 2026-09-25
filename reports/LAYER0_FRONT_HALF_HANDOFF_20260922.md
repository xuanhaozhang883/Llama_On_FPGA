# 第 0 层前半层代码交付验证

日期：2026-09-22。本文验证代码准备能力，不表示最终接口冻结、目标模型联调或实板验收完成。

## 已完成的代码工作

- 提交此前本地的独立 Tokenizer、Embedding、流式 safetensors 读取、Embedding 校验和可选 INT8 实验工具。
- QKV 同时输出 input RMSNorm 前的 `residual_hidden_fp32.npy`，FP32 `[1,L,4096]`。
- QKV 保持 BF16 小端 `[head,token,dim]` 布局、128 行及有效长度后的补零规则；输出文件关联 CRC32/SHA-256。
- Metadata 记录 config、权重索引、Tokenizer 哈希、可用模型包身份及计算精度。路径仅作来源说明，可跨电脑使用。
- 联调默认 CPU / FP32 / 单线程，保留显式选择其他计算方式的实验入口。
- PowerShell 明确要求模型目录，允许分别指定计算与 Tokenizer 解释器，不依赖固定的本机虚拟环境。
- 新增固定官方模型 revision、按权重索引选完整第 0 层分片、逐文件校验和生成交付清单的工具。
- 输出重写前撤销旧完成 Manifest，避免失败后旧清单误报当前输出完整。

## 验证结果

运行环境包含 PyTorch `2.4.1+cpu`、NumPy `1.24.4`、safetensors `0.5.3` 和 huggingface_hub `0.36.2`。计算解释器为本机 `D:\ANACONDA\python.exe`。

```powershell
D:\ANACONDA\python.exe -m unittest tests.test_prepare_llama3_qkv tests.test_prepare_llama31_model -v
```

结果：**15 项测试全部通过**。

覆盖：合成权重的投影、Residual 的时点与形状、布局与补零、BF16 舍入、直接与分步流水线一致性、跨目录产物读取、Tokenizer/index/bundle 身份不匹配、部分覆盖失败的旧清单失效，以及模型获取流程的固定 revision、分片选择、校验失败、旧配置拒绝和中断恢复。

另完成：

- `scripts/run_embedding.ps1` 的 PowerShell 语法检查通过。
- 使用本机旧 `D:\Llama_weight` 跑通分词、Embedding 导出及逐行比对。输入“你好”对应 `[128000,57668,53901]`，输出 `[3,4096]` FP32；计算和分词分别使用显式指定的解释器。
- 使用旧模型和 `[128000,114880,48044,127392]` 导出完整板级尺寸 QKV，验证每个文件的长度、CRC32、SHA-256 和补零区域；Residual `[1,4,4096]` 与此前同输入的 Embedding 精确相等。
- `git diff --check` 通过。

旧模型冒烟产物存放于被 Git 忽略的 `out/handoff-smoke-legacy/`。它们明确不是 Llama 3.1 Instruct 验收数据，不能拿来冻结新 RoPE ROM 或目标模型数值基线。

## 未完成与阻碍

目标保持为官方 `meta-llama/Llama-3.1-8B-Instruct`，本次解析 revision 为 `0e9e39f249a16976918f6564b8830bc894c89659`。本机已有 Hugging Face 登录，但请求该 revision 的 `config.json` 返回 HTTP 403，服务器说明该账户不在获准访问名单内。

因此尚未生成正确目标模型的完整文件包及本地 SHA-256 清单，也未完成该模型的真实前半层数值验收。需在本机完成官方访问授权／切换到已获授权账户，或提供同源、可校验的已下载模型目录。现有旧模型未被覆盖，RoPE ROM 未替换。

正式 Attention TCP 服务器、最终 XSA/BSP、PHY 实测配置、链接脚本、ELF/Map、可复现服务器构建和 Fail-stop 实板验证仍属于后续实现交付。当前提交仅确认由我方承担这些责任，并明确链接布局排除 `[0x10000000,0x10280000)`。

本机存在 Vivado/Vitis 2025.2 安装目录 `E:\vivado_25_2\2025.2`，但工具的存在不代表本次已经构建或验证了服务器工程。接口冻结与后续工程验收流程见 `docs/layer0_handoff_contract.md`。
