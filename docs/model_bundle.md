# Llama 3 Base第0层模型文件检查与交付

当前唯一目标是官方`meta-llama/Meta-Llama-3-8B` Base。当前里程碑只验证decoder第0层，不使用Instruct聊天模板，也不需要`chat_template`。

本项目区分两种检查结果：

- **本地冒烟审计**：确认朋友拷贝来的文件能否支持第0层计算，但不能证明文件来自哪个官方revision。
- **正式模型交付**：从官方仓库固定到40位commit SHA，校验Hub提供的Git/LFS摘要，再生成状态为`complete`的Manifest。

## 检查当前本地目录

在工作树根目录执行：

```powershell
.\.venv\Scripts\python.exe python\prepare_llama3_model.py `
  --local-model-dir "C:\lhm\2_Work\Llama3-8B" `
  --output-dir "out\llama3-model-audit"
```

结果写入`out/llama3-model-audit/local-model-audit.json`。在没有官方revision和远端摘要时，状态必须是`identity_unverified`。

报告分别记录：

- `layer0_assets_complete`：Embedding和第0层所需文件是否齐全、Safetensors结构是否完整；
- `full_model_files_complete`：索引引用的全部32层分片是否都在本地；
- `missing_full_model_shards`：缺少的完整模型分片；
- 本地配置、索引、Tokenizer和第0层所需分片的大小及SHA-256。

第0层全部张量位于第1分片，只能说明第0层可以测试。当前缺少第2、3分片，因此不能宣称完整32层模型可用。

## 将来完成正式交付

获得官方受限仓库访问权限后执行：

```powershell
.\.venv\Scripts\python.exe python\prepare_llama3_model.py `
  --output-dir "out\models" `
  --revision main
```

程序首先将`main`解析成不可变的40位commit SHA，然后下载同一revision的配置、Tokenizer、权重索引和第0层所需分片。所有文件通过远端Git blob或LFS SHA-256检查后，才会原子写入`model-bundle-manifest.json`并标记`complete`。

只检查元数据而不下载权重时可以增加`--metadata-only`。这只会生成`bundle-plan.json`和`metadata_only_weights_pending`状态，不能作为正式完成证明。

## 配置边界

目标配置必须满足：

- 32层、hidden 4096、32个Q头、8个KV头、head_dim 128；
- intermediate 14336、vocab 128256、BF16权重；
- `max_position_embeddings=8192`；
- `rope_theta=500000`；
- `rope_scaling`与`rope_parameters`缺失或为`null`；
- 不使用Attention或MLP bias。

模型检查通过不等于RoPE ROM已经签核。候选LUT与当前ROM的差异必须单独调查，未经确认不得覆盖`mem/sin_bf16.hex`或`mem/cos_bf16.hex`。

## 软件测试

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prepare_llama3_model -v
```

测试使用本地小型fixture，不访问真实Hugging Face仓库，也不读取16 GB模型。

参考：[Meta-Llama-3-8B官方模型仓库](https://huggingface.co/meta-llama/Meta-Llama-3-8B)、[Hugging Face固定版本下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)。
