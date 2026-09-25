# Llama 3.1 → Llama 3 目标模型迁移设计

> 日期：2026-09-24
> 目标：将第0层 PC–FPGA–PC 闭环的唯一目标，从
> `meta-llama/Llama-3.1-8B-Instruct` 改为
> `meta-llama/Meta-Llama-3-8B` Base模型。

## 1. 已确认目标

- 模型仓库：`meta-llama/Meta-Llama-3-8B`。
- 模型类型：Base预训练版，不使用Instruct聊天模板。
- 当前里程碑：只验证第0个Transformer层。
- 硬件序列长度继续固定为128，batch继续固定为1。
- RoPE继续由FPGA执行，PC输出RoPE前Q/K/V。
- Llama 3 RoPE为`rope_theta=500000`、无scaled-RoPE；模型最大上下文为8192。

本地模型目录固定为：

```text
C:\lhm\2_Work\Llama3-8B
```

本地证据已经确认：

- 配置为32层、hidden 4096、32Q/8KV、intermediate 14336；
- `max_position_embeddings=8192`；
- `rope_theta=500000`；
- `rope_scaling=null`；
- 第0层前后半程所需10个权重全部位于第1分片；
- 第1、4分片按Safetensors头部和数据偏移检查结构完整；
- 第2、3分片缺失，因此当前只允许第0层测试，不能宣称完整32层模型可用。

准确下载revision和官方哈希尚未获得。该缺口不阻塞本地冒烟测试，但阻塞正式模型身份签核。

## 2. 选择的迁移方案

采用“单一目标、完整纠正”方案：仓库只把`Meta-Llama-3-8B`视为当前正式目标，删除3.1专用验收语义，同时保留模型无关的已完成模块。

没有选择以下方案：

1. **只替换文档字符串**：改动最小，但3.1 scaled-RoPE检查仍会错误拒绝真实Llama 3配置。
2. **同时支持Llama 3与Llama 3.1**：更通用，但当前项目只有一个硬件目标，会增加配置分支和验收组合。

## 3. 保持不变的模块

以下接口和实现不因目标切换而改变：

- Q：BF16小端`[32,128,128]`；
- K/V：BF16小端`[8,128,128]`；
- Context：BF16小端`[32,128,128]`；
- `residual_hidden`：input RMSNorm前的CPU FP32`[1,L,4096]`；
- TCP v1头部、CRC、状态码和端口；
- DDR地址和Cache职责；
- GQA头映射、因果Mask和Context拼头；
- PC后半层`o_proj → Residual → RMSNorm → SwiGLU → Residual`；
- FPGA RoPE乘加与split-half旋转数据通路。

已经完成的协议、PC前半层、Attention参考和PC后半层测试继续作为回归门禁。

## 4. 必须迁移的内容

### 4.1 模型交付与验证

- 将3.1专用的模型准备/验证模块改为Llama 3专用模块；
- 仓库ID改为`meta-llama/Meta-Llama-3-8B`；
- 配置验收改为8192上下文、`rope_theta=500000`、无scaled-RoPE；
- 保留32层、4096 hidden、32Q/8KV、14336 intermediate、无Attention/MLP bias等结构检查；
- 正式Manifest必须记录repo ID、解析后的40位revision和逐文件SHA-256；
- revision未知时，允许产生本地冒烟报告，但状态必须明确为`identity_unverified`，不得伪装为正式完成。

文件和测试名称使用`llama3`，不继续使用`llama31`命名。旧名称只允许在Git历史中存在。

### 4.2 RoPE资产

- 新生成器按照普通Llama 3公式生成position 0～127、64个频率的sin/cos；
- BF16转换必须使用round-to-nearest-even；
- 生成结果先写入临时输出目录，不直接覆盖`mem/*.hex`；
- 与独立PyTorch参考逐项比较16,384个字；
- 当前ROM与PyTorch式普通Llama 3参考存在4个BF16字差异，必须先定位计算顺序或舍入来源；
- 在差异来源明确前，不改ROM、不改Golden、不重建bitstream；
- 如果官方参考与现有硬件资产无法同时满足，立即停止并由用户选择迁移策略。

### 4.3 文档

更新设计、实施计划、模型交付说明和接口说明中的：

- 模型名称与仓库ID；
- Base而非Instruct的输入语义；
- 8192上下文；
- 普通RoPE而非Llama 3.1 scaled-RoPE；
- 本地分片可用范围和正式身份验收边界。

TCP字段、张量字节数、DDR地址和Fail-stop语义不改。

## 5. 测试策略

迁移使用TDD：

1. 先把模型验证测试改成Llama 3期望，并观察3.1实现失败；
2. 再修改模型验证实现使测试通过；
3. 为普通RoPE生成器先建立独立小型和完整8192项测试；
4. 对本地模型目录执行只读配置、索引和第0层张量检查；
5. 运行PC前半层和后半层的短Token冒烟测试；
6. 最后运行全部Python测试，任何3.1残留验收逻辑都视为失败。

当前ROM差异调查与ROM替换分成两个门禁，调查完成不等于授权替换。

## 6. 完成条件

目标迁移完成必须同时满足：

- 代码和现行文档只把`Meta-Llama-3-8B`作为正式目标；
- 模型验证接受真实Llama 3配置并拒绝Llama 3.1 scaled-RoPE配置；
- 本地第0层权重检查通过，同时明确第2、3分片缺失不支持32层；
- PC前半层、Attention参考和PC后半层回归通过；
- Llama 3 RoPE生成器与独立参考逐字一致；
- 当前ROM的4字差异已有证据化结论；
- 未获得revision/官方哈希时，正式模型身份保持`NOT RUN`或`identity_unverified`；
- 没有修改FPGA RoPE数据通路RTL。
