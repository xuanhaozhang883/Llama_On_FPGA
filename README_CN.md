# Llama Attention FPGA

这是 Llama3 风格 Attention 层的 FPGA 实现，目标器件为
`XCZU15EG-FFVB1156-2-I`。当前工程处理 DDR 中已经生成的 Q、K、V，
完成 RoPE、QK、因果掩码、在线 Softmax、与 V 融合以及 Context 写回。

它还不是完整的 Llama3-8B 推理系统。电脑端已有独立 Embedding 步骤和
第 0 层 RMSNorm、QKV 投影生成工具；板上尚未实现这些算子、输出投影、
MLP/SwiGLU、Residual、KV Cache、32层调度、LM Head 和 Token 采样。

## 当前状态（2026-08-30）

- 当前生产主线：32个 RTL 文件、33个模块。
- 当前回归：13个 SystemVerilog TB；Mac 端 Icarus 和 Python 数值回归通过。
- 历史发布包曾通过 XCZU15EG 实板验证。
- 当前 `unsigned_restoring_divider.sv` 与历史发布版不同；修改版已经通过
  Icarus 测试，但尚未重新完成 Vivado 2025.2 全流程和实板验证。
- 因此当前源码不能宣称与历史 BIT/XSA 完全一致。

## 目录

| 路径 | 内容 | 是否日常修改 |
|---|---|---|
| `rtl/board/` | 板级顶层、AXI、DDR读写和调度 | 板级接口变化时修改 |
| `rtl/core/` | 当前 Attention 计算主线 | 算法/架构开发主要修改处 |
| `tb/` | 当前 RTL 单元和集成测试 | 修改 RTL 时同步增加测试 |
| `tests/` | PowerShell/Vivado 测试入口和 Stub | 一般只运行 |
| `scripts/` | Vivado、Vitis、上板 Tcl | 工具链或板卡变化时修改 |
| `mem/` | RoPE 和指数 LUT 初始化文件 | 改算法/量化格式时生成 |
| `bd_base/` | Zynq PS、DDR、时钟 Block Design | 换板或改 PS 配置时修改 |
| `python/` | 数值模型、Golden 数据和日志分析 | 算法验证时修改 |
| `vitis/` | ARM 端上板测试程序和输入/Golden 数据 | 上板测试时修改 |
| `archive/` | 原始发布包和旧版 RTL/TB | 只读参考，不加入工程 |
| `reports/` | 验证报告；脚本可重新生成 | 不手工维护结果文件 |
| `export/` | Vivado 构建后生成的 XSA | 生成目录，目前可不存在 |

## RTL 连接关系

```text
attention_board_top
├─ design_1_wrapper                 Zynq PS / DDR / 时钟（由 BD 生成）
├─ aq_axi_master_fixed              PL 侧 AXI Master
└─ fpt_attention_board_engine       板级调度
   ├─ fpt_raw_qk_ddr_reader         读取 Q/K
   ├─ fpt_v_ddr_loader              读取 V
   ├─ flash_attention_system_with_rope_top
   │  └─ rope_qk_flash_attention_pipeline_top
   │     ├─ RoPE
   │     └─ qk_flash_attention_pipeline_top
   │        ├─ QK 脉动阵列和因果跳过/掩码
   │        └─ flash_attention_consumer_top
   │           ├─ Online Softmax
   │           ├─ V Cache
   │           └─ Context Fusion
   └─ fpt_context_ddr_writer        写回 Context
```

板级入口是 `rtl/board/attention_board_top.sv`。Vivado 使用的完整生产文件
清单在 `scripts/source_manifest.tcl`，新增生产 RTL 后必须显式加入该清单。

当前因果掩码不是一个独立顶层：完整无效 Tile 在 QK 阵列处跳过，Tile 内
无效元素由 `qk_flash_attention_pipeline_top.sv` 生成 mask 元数据。

## 在 Windows 上验证和构建

建议使用 Vivado/Vitis 2025.2，并先执行对应的 `settings64.bat`。

1. 运行当前仿真和数值回归：

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tests\run_v313_qk4_system_checks.ps1
   ```

2. 只做整板 RTL Elaboration：

   ```bat
   01_check_rtl.bat
   ```

3. 综合、实现、生成 BIT/XSA：

   ```bat
   02_build_bitstream.bat
   ```

4. 构建 Vitis 裸机测试程序：

   ```bat
   03_build_vitis.bat
   ```

5. 上板前按 `scripts/run_on_board_no_gtr_xsct.tcl` 文件开头的说明提供
   `psu_init.tcl` 和应用 ELF。串口参数为 115200 8N1。

Vivado 的生成工程默认放在源码目录旁的 `_fpt_v313_build/`，不应提交或
复制回 `rtl/`。

## 第 0 层前半层交付（2026-09-22）

共同目标固定为 `meta-llama/Llama-3.1-8B-Instruct`。当前交付包含 PC
端 Tokenizer、Embedding、input RMSNorm、RoPE 前 Q/K/V，以及
`residual_hidden_fp32.npy`；正式 Attention TCP 服务器仍待双方接口冻结后
实现。责任划分、Fail-stop 和 DDR 预留约定见
[交付约定](docs/layer0_handoff_contract.md)。这次代码同步不是最终接口冻结。

### 获取同源模型

本机原有 `D:\Llama_weight` 是旧 Llama 3 配置，不能代替目标模型。
正确的 config、Tokenizer、索引和第 0 层所需分片必须来自同一个官方 commit。
由前半层／服务器负责人准备交付包及逐文件 SHA-256；获取方法见
[模型交付说明](docs/model_bundle.md)。模型文件保存在已忽略的 `out/models/`
或仓库外，不提交 Git。

使用获得官方仓库访问授权的 Hugging Face 账户在本机登录，不把令牌写入
仓库或命令行参数。建议使用 Python 3.10+ 的独立环境：

```powershell
python -m pip install numpy torch safetensors transformers huggingface_hub
hf auth login
python python/prepare_llama31_model.py --output-dir out/models --revision 0e9e39f249a16976918f6564b8830bc894c89659 --metadata-only
python python/prepare_llama31_model.py --output-dir out/models --revision 0e9e39f249a16976918f6564b8830bc894c89659
```

该 revision 为本次从官方仓库解析的完整 commit。脚本按索引选取 Embedding
和完整第 0 层所需分片，不假设分片编号。完整下载并校验后才生成
`model-bundle-manifest.json`；`bundle-plan.json` 只是元数据计划。
本次访问目标 config 得到 403 未授权，因此尚未获得目标模型交付包，不能
声称真实 Llama 3.1 数值验证通过。双方核验新模型和 Golden 前不替换 RoPE ROM。

### 生成 Token IDs、Embedding、QKV 和 Residual

下面的解释器应安装上述依赖。将 `$modelDir` 指向完整交付包，先固定输入，
双方后续联调使用生成的同一份 Token IDs，不各自重新分词。

```powershell
$modelDir = Resolve-Path './out/models/Llama-3.1-8B-Instruct-0e9e39f249a16976918f6564b8830bc894c89659'
python python/tokenize_llama3.py --model-dir $modelDir --text '你好' --output out/embedding/token_ids.json
python python/prepare_llama3_embedding.py --model-dir $modelDir --token-ids-file out/embedding/token_ids.json --output-dir out/embedding
python python/verify_llama3_embedding.py --model-dir $modelDir --embedding-dir out/embedding
python python/prepare_llama3_qkv.py --model-dir $modelDir --token-ids-file out/embedding/token_ids.json --embedding-dir out/embedding --device cpu --compute-dtype fp32 --num-threads 1 --output-dir out/qkv_layer0
```

也可以用 `scripts/run_embedding.ps1` 串联前三步，必须明确模型目录；
`-TokenizerPython` 可指定独立分词环境，省略时复用 `-ComputePython`。
例如在本机已安装的两个环境中：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_embedding.ps1 -Text '你好' -ModelDir $modelDir -ComputePython D:\ANACONDA\python.exe -TokenizerPython .\.venv-tokenizer-clean\Scripts\python.exe
```

脚本按磁盘偏移读取 safetensors 所需行／张量，不映射整个多 GB 分片。
`--token-ids` 可直接指定 ID（不会自动添加 BOS）；输入长度要求 1–128，
超过长度直接报错。`--text` 使用本地 Tokenizer，不联网。

| 输出 | 类型和形状 | 用途 |
|---|---|---|
| `q_before_rope_bf16.bin` | BF16 小端 `[32,128,128]`，1,048,576 字节 | Q，DDR `0x10000000` |
| `k_before_rope_bf16.bin` | BF16 小端 `[8,128,128]`，262,144 字节 | K，DDR `0x10100000` |
| `v_bf16.bin` | BF16 小端 `[8,128,128]`，262,144 字节 | V，DDR `0x10140000` |
| `residual_hidden_fp32.npy` | FP32 `[1,L,4096]` | RMSNorm 前 Embedding，留在 PC |
| `manifest.json` | JSON | 同一输入的模型身份、Token IDs、文件哈希、CRC、精度和布局 |

QKV 布局为 `[head][token][dim]` 连续存储，投影后第 L 行起补零；PL 仍按
固定 128 行计算，只有前 `valid_tokens=L` 行用于真实模型验收。Residual
不补零，也不发送给 PL。BF16 输出使用 round-to-nearest-even。

默认 CPU / FP32 / 单线程，Manifest 记录实际 PyTorch/NumPy 版本、RMSNorm
和投影精度。仍可显式选择 `--device auto|cuda` 或 `--compute-dtype bf16`，
但这些是独立实验，不与首版基线混用。即使固定 CPU 配置，不同软件／CPU
也不能先验保证完全相同的浮点结果；联调优先共享已校验的同一份二进制。

新 Token IDs、Embedding 和 QKV metadata 记录 config、权重索引、Tokenizer
哈希，并关联可用的模型交付清单。模型绝对路径仅作来源说明，可跨电脑移动。
小文件哈希和清单身份校验不会每次重新扫描全部权重；模型分片应在交付时
完成 SHA-256 校验并保持不变。旧 Manifest 兼容仅保留其原有哈希约束，不用作
正式接口冻结的数据。

`python/quantize_llama3_embedding.py` 与 `--int8-dir` 保留为可选实验。
量化会改变 Embedding；该格式尚不提供完整源权重／量化数组内容认证，不用于
首版正式 BF16 模型验收。

### 验证和当前边界

```powershell
python -m unittest tests.test_prepare_llama3_qkv tests.test_prepare_llama31_model -v
```

测试使用小模型覆盖投影、Residual、布局、padding、BF16 舍入、跨目录交付、
模型身份及获取流程；不等于真实目标模型或实板验收。
当前 `vitis/eth_echo` 仍只做回显，`vitis/src/fpt_attention_board_test.c` 从
编译进程序的 Golden 加载 DDR，尚未形成 QKV TCP 请求到 Context 响应的正式服务器。

## 修改规则

1. 改 RTL 前先确认它在 `scripts/source_manifest.tcl` 的生产清单中。
2. 每次 RTL 修改至少运行相关 TB，再运行完整系统检查。
3. 端口或层级变化后必须运行 `01_check_rtl.bat`。
4. 不要从 `archive/` 直接引用 RTL；需要恢复时先复制回生产目录并补测试。
5. 新增完整 Llama 模块时按功能新建清晰目录，不再使用 `a`、`bc` 这类历史命名。

## 尚未解决

1. 在 Vivado 2025.2 中重新完成 Elaboration、Synthesis、Implementation、
   Timing/DRC、Bitstream 和 XCZU15EG 实板验证。
2. 确认修改后的 `unsigned_restoring_divider.sv` 是否作为正式实现保留。
3. 将 `rtl/core/a`、`rtl/core/bc` 重命名为可读的功能目录；这要等 Vivado
   基线通过后再做，避免一次同时改变内容和路径。
4. 为完整 Llama3-8B 逐步增加线性层、RMSNorm、MLP、Residual、KV Cache、
   多层控制和 Token 输出链路。
