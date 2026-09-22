# Llama 3.1 第 0 层模型交付包

目标固定为官方 `meta-llama/Llama-3.1-8B-Instruct`。由前半层及服务器负责人准备并交付同源 config、Tokenizer、权重索引、Embedding 与完整第 0 层所需分片，以及逐文件 SHA-256。旧 `D:\Llama_weight` 不作为该目标的来源，也不由此脚本修改。

需要已安装 `huggingface_hub`，并使用已经获准访问该仓库的 Hugging Face 登录。脚本复用本机认证，不接收或打印 token。请勿将 token 写入命令行、仓库或交付文件。

## 先检查元数据

在仓库根目录执行：

```powershell
D:\ANACONDA\python.exe python/prepare_llama31_model.py --output-dir out/models --revision main --metadata-only
```

脚本首先把 revision 解析为完整的 40 位 commit SHA，后续所有文件均从该 commit 获取。它下载 config、Tokenizer 和索引等小文件，验证 Llama 3.1 8B 的结构及 `llama3` scaled-RoPE 参数；根据索引找出 Embedding 和第 0 层的全部分片。不会假设所需权重都在第一个分片。

输出在 `out/models/Llama-3.1-8B-Instruct-<完整commit>/bundle-plan.json`。这个文件仅为计划，状态为 `metadata_only_weights_pending`，其中权重分片尚未获得本地 SHA-256，不能当作完成交付。

## 下载分片并完成清单

使用计划中记录的完整 commit 替换下面占位符：

```powershell
D:\ANACONDA\python.exe python/prepare_llama31_model.py --output-dir out/models --revision <完整commit>
```

同样的命令可以重复执行，Hugging Face 库复用缓存并支持中断后继续下载。输出与旧模型隔离在独立的 commit 目录；发现目录中不属于本次交付的文件时拒绝继续。

脚本流式计算每个文件的 SHA-256，并校验 Hub 提供的 Git blob ID 或 LFS SHA-256。所有所需文件通过后，原子写入 `model-bundle-manifest.json`；失败时不会生成新的完成清单，重新验证前会撤销当前目录原有的完成清单。遇到本地文件损坏，可以添加 `--force-download` 重新获取。

清单包含来源 repo、固定 revision、完整 config、RoPE 字段、逐 tensor 到 shard 的映射、逐文件大小和 SHA-256。它只保证 Embedding 及完整第 0 层所需的交付，不表示已经下载完整的 32 层模型。获得正式清单并由双方核验前，不替换 RoPE ROM。

## 验证脚本

```powershell
D:\ANACONDA\python.exe -m unittest tests.test_prepare_llama31_model -v
```

测试使用小型本地 fixture 覆盖 revision 固定、索引分片选择、远端及本地哈希校验、旧模型配置拒绝、下载失败及恢复，以及无关文件隔离；无需下载实际模型。

参考：[官方模型仓库](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)、[Hugging Face 固定版本下载及本地缓存说明](https://huggingface.co/docs/huggingface_hub/guides/download)。
