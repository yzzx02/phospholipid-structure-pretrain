# 迁移交接：RTX 4070，原生 Windows，现有默认 CUDA 环境

用户确认后续使用 RTX 4070 那台电脑的原生 Windows 和已经在用的默认 CUDA 环境。
本次只整理项目与数据，不指定或修改 Python、PyTorch、CUDA、驱动版本，不复制 Linux Conda 环境。
当前 AMD 主机的 GPU 作业已因黑屏停止，没有自动重试。

## 带走什么

按用户要求采用精简包：保留运行必需的冻结输入、两份最终权重与最新项目资料。
复制整个迁移目录，包括所有 `.zip.001`、`.zip.002` 等分卷和旁边的 JSON、校验文件、说明、恢复脚本。
分卷每个最多 2 GiB，便于不同硬盘文件系统保存。不要只复制第一个分卷。

ZIP 内的根目录为 `phospholipid-structure-pretrain/`，包含：

|内容|路径|用途|
|---|---|---|
|已提交源码、配置、测试、报告|仓库全部 tracked 文件|恢复同一代码版本与实验结论|
|冻结的 2,015,000 条结构库|`data/structure_library_v0.3/`|固定甘油/链/Polar 库和 Builder 数据|
|v0.3 必需标签与分片|`data/private/pretraining_v03/` 的核心文件和 `rows/`|完整式、链标签、Polar 标签、Tokenizer、划分、真实数据 shards|
|v0.4 核心数据与缓存|`data/private/pretraining_v04/` 的核心文件和 `fragments/`|无链特征、公式统计、已完成的 H/HC 检索缓存|
|Morgan 3,000 步权重|`outputs/pretraining_v04/runs/fp_smoke_3000/checkpoint/step_003000.pt`|含模型与续训状态|
|GINE 3,000 步权重|`outputs/pretraining_v04/runs/gine_smoke_3000/checkpoint/step_003000.pt`|含模型与续训状态|
|固定样本池|`outputs/pretraining_v04/samples/`|20k train、2k validation、512 诊断、32 样本固定索引|
|故障摘要|两个 GPU 停止锁及 `gpu_safety_final.json`|保留本机故障事实|
|逐文件 SHA256|`PROJECT_TRANSFER_MANIFEST.json`|核对每个文件内容与源 commit|

旧 v0.3 训练输出、过拟合/预检输出、中间 checkpoints、旧检索 bank、索引构建中间文件、
逐样本历史预测和 Git 历史不打包。原机器上的历史文件仍保留，没有删除或覆盖。
现有 H/HC 缓存直接供 v0.4 使用；该精简包不以重跑旧建库/建索引流程为目标。

私有数据与检查点不在 GitHub，需要从硬盘带走。导出没有包含 `.git` 凭据、用户目录或 ROCm 环境。

## 在 Windows 解包与核验

例如迁移目录在 `E:\phospholipid_rtx4070_transfer_20261020`，准备解包到新的 `D:\phospholipid_workspace`。
替换成实际盘符；目标父目录中不能已有 `phospholipid-structure-pretrain` 或同名合并 ZIP。

在 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "E:\phospholipid_rtx4070_transfer_20261020\RESTORE_WINDOWS.ps1" `
  -PackageDirectory "E:\phospholipid_rtx4070_transfer_20261020" `
  -DestinationParent "D:\phospholipid_workspace"
```

脚本依次核对分卷大小和 SHA256，合并 ZIP64，核对合并包哈希，使用系统 `tar.exe` 解压，
最后核对全部文件的大小和 SHA256。它不安装包、不启动模型，也不改动现有 CUDA 环境。
解包时还需为合并 ZIP 和展开数据预留空间；精确大小在 `PACKAGE_RECEIPT.json` 中。
原始分卷保持原样，以便重新核验或再次迁移。

解包成功后从新的项目根目录工作，先阅读本交接文件与下列实验报告。
`PROJECT_TRANSFER_MANIFEST.json` 的 `source_commit` 是包内源码的准确版本；
GitHub 后续的迁移回执可能是单独的文档提交，以回执记录的包内 commit 为准。

## 已完成与未完成

- v0.4 Morgan/FP 和 GINE 均已有 3,000 步权重；本轮没有重新训练。
- 358 项 CPU 测试通过；512 个 validation 目标的完整式（含 H）与完整图往返重建全部通过。
- 188 个目标有多个 P，130 个目标的连接 P 不是 SMILES 中第一个 P。继续保留内部 `[*:1]` 标签。
- Beam 5 的两个 encoder、四方法完整比较已完成。
- Morgan Beam 20 只完成普通 Beam 和同批候选的后验过滤。
- Morgan Beam 20 元素约束在 241/256 目标后中断；该记录不属于完整 512 目标对照。
- 其余 Beam 20 与正式 Beam 50 没有执行。GINE Beam 50 只有两目标预检，不算正式实验。
- 2026-10-20 10:44，当前 Windows/WSL/ROCm 组合新增 watchdog，WER `LiveKernelEvent 141`；用户确认黑屏后恢复。
- 精确故障根因未定位。所有停止锁、原始日志和新 dump 的 SHA256 已保留；未重新启动 GPU 工作。

详见 [本轮结果](results/POLAR_FORMULA_BEAM_V04_RESULTS.md)、
[聚合 JSON](results/POLAR_FORMULA_BEAM_V04_RESULTS.json)、
[方法和原复现参数](experiments/POLAR_FORMULA_BEAM_V04.md)。

Beam 5 的“重原子＋基础语法＋最终检查”把严格分子式覆盖率从 11.72%/9.38% 提高到 38.09%/30.27%，
但 Polar Top-5 exact 仅 8/512 和 5/512。公式正确不等于目标结构正确。
该模型使用 structure-oracle 模板检索、可信式与已知链，不能把这些数值当作 MS/MS 鉴定准确率。

## 接续机器需要处理的代码事项

**数据包就绪不代表 NVIDIA 原生 Windows 入口已经验证。** 本次没有接入 4070，也没有修改其环境。
以下是后续接续工作的明确清单，不是本次要自动执行的任务：

1. 先用现有默认解释器确认项目文件与必要依赖能导入、CPU 测试通过；记录现有环境信息。
2. 在新机器核实 CUDA 与设备名称，再建立独立的 NVIDIA 作业输出目录，保留此 AMD 实验与停止记录。
3. 当前 `polar_only_v04/runtime.py::require_gpu` 检查 HIP 和 RX 9070 XT，不能直接用于 NVIDIA。
   新的 NVIDIA 入口需单独接入 CUDA 检查，保留旧 ROCm 路径；不得因此修改模型架构或权重。
4. 当前 `scripts/run_rocm.sh` 和两个 ROCm PowerShell 监督器依赖 WSL；不要把它们当作原生 Windows 启动器。
5. `scripts/check_formula_beam_v04.py` 的解释器路径固定为当前 WSL；
   `formula_beam_eval.py` 的 RAM 统计使用 Linux `resource`。新入口需处理 Windows 路径和资源统计。
6. 保持原有 tokenizer、`[*:1]`、Builder、数据划分、检索缓存和固定 512 头基，不重建标签。
7. 先运行小规模 CPU/设备检查和有界推断，复核 Beam 5 基线，再补齐固定 512 目标的 Beam 20/50 对照。
   不把 AMD 中断前缀 241 条拼成另一机器的完整运行；每台机器各自单独报告耗时和资源。
8. 本轮没有授权启动 5,000 步、50k 训练池、201.5 万全量预训练或 MS/MS 微调。

4070 后续使用现有默认环境；不根据本交接说明自动重装驱动、切换系统或批量升级依赖。
