# v0.4 Polar-only：原生 Windows / RTX 4070 SUPER 接续

本入口复用迁移包中的 Morgan 与 GINE 3,000 步权重，接续分子式约束推断。
保留原 ROCm 入口、停止锁及结果；NVIDIA 使用独立 epoch、CPU gate 和输出目录。
模型架构、Tokenizer、内部 `[*:1]`、Builder、冻结结构库、划分和检索缓存均保持原样。
没有授权或启动新的训练、CL、50k 池、全量预训练或 MS/MS 微调。

## 本机布局

- 工作根目录：`G:\yzx\linzhiyuce`
- Git 仓库：`phospholipid-structure-pretrain`
- 原迁移包：`phospholipid_rtx4070_transfer_20261020`
- 逐文件校验后的恢复副本：`restored_v04\phospholipid-structure-pretrain`
- Python：`environments\polar-v04-cu128\Scripts\python.exe`
- 下载、临时文件和模型缓存：`cache\pip`、`cache\tmp`、`cache\wheels`、`cache\torch`、`cache\huggingface`
- 本机实验：`phospholipid-structure-pretrain\outputs\formula_constrained_v04_rtx4070_20261020`

ZIP SHA256 为 `4414514aba55dc6beab251d720a7f2ba4f94e9a547d84c8ba796dcc5f2698de3`，
775 个文件的大小和 SHA256 全部通过；包内源码为
`c5d56cf86040a54062b40fa9589994f98c754c00`。
GitHub 克隆起点为 `f3969d1`，新增内容是迁移回执。

仓库中的 `data/private`、`data/structure_library_v0.3`、`outputs/pretraining_v04`
以 Windows junction 指向已校验恢复副本的同名路径，避免再复制冻结数据。
**不要删除或移动 `restored_v04`；这些 junction 仍依赖它。**
推断结果写入独立目录，不写入上述冻结输入和旧 checkpoint 目录。

## 环境与安装

本机为 Windows、RTX 4070 SUPER 12 GB、NVIDIA 581.08、32 GB RAM。
驱动没有重装或升级。原默认 Python 没有项目依赖；找到的旧 venv 指向其他机器的解释器，
所以在 G 盘建立独立 Python 3.12.10 环境，没有修改系统默认环境。

核心版本：PyTorch `2.9.1+cu128`、torch-geometric `2.8.0.post1`、
RDKit `2026.03.3`、NumPy `2.2.3`，其余按 `constraints-tested.txt` 固定。
CUDA wheel 随 PyTorch 安装；采用 [PyTorch 官方安装方式](https://pytorch.org/get-started/locally/)。
CUDA 版与原 ROCm 版采用同一 PyTorch 基础版本，权重是否兼容仍以真实加载和推断为准。

仓库根目录执行下面命令可重建同类环境；先将临时目录和 pip cache 指向 G 盘。

```powershell
$env:TEMP='G:\yzx\linzhiyuce\cache\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR='G:\yzx\linzhiyuce\cache\pip'
py -3.12 -m venv ..\environments\polar-v04-cu128
$python='..\environments\polar-v04-cu128\Scripts\python.exe'
& $python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -c constraints-cuda-v04.txt -r requirements-extraction.txt -e '.[dev,library,retrieval]'
```

本次官方索引指向的 `download-r2.pytorch.org` 下载曾停滞，实际改用同版本官方文件：
`https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl`。
官方响应中的 SHA256 为 `3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0`。
只对当前子进程使用 Windows 已配置的本地 HTTP 代理，没有更改系统代理或 Git 全局配置。

以后每次打开 PowerShell，执行：

```powershell
. G:\yzx\linzhiyuce\phospholipid-structure-pretrain\scripts\enter_cuda_v04.ps1
```

这会切换到项目根目录、选择独立解释器，并将进程级的临时目录、pip、Torch 与 Hugging Face 缓存放在 G 盘。
无需激活其他项目的旧 venv。

## 验证顺序

先执行 CPU gate；关闭 GPU 的子进程运行全套测试、pip check、两份真实 checkpoint 的 512 标签审计，
以及全部 512 目标通过原 Builder 的完整式（包括 H）和完整图重建。

```powershell
python scripts/check_formula_beam_v04.py --gpu-backend native_cuda `
  --experiment-root outputs/formula_constrained_v04_rtx4070_20261020
```

完成后可用 `--verify-existing` 只读复核 gate。已有 gate、审计和运行目录均不覆盖。
原始 ROCm CPU gate 默认仍检查原固定解释器。

先以两目标预检真实模型和高 beam，再扩大到固定 512 的子集。
`preflight_` 前缀的作业单独报告，不混入正式目标聚合。

```powershell
python scripts/supervise_formula_beam_cuda_v04.py --encoder fingerprint `
  --mode elements_grammar --beam-size 5 --count 2 --run-name preflight_fp_b5
python scripts/supervise_formula_beam_cuda_v04.py --encoder graph `
  --mode elements_grammar --beam-size 50 --count 2 --run-name preflight_gine_b50
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_formula_beam_cuda_v04.ps1 -Count 32
```

启动器串行执行两 encoder、Beam 5/20/50 和三种搜索；普通 Beam 同批候选的后验过滤构成第四种方法。
每个作业限制 1,200 秒；初始化允许 180 秒，开始解码后 20 秒无进展停止。
普通 Beam 只增加监控心跳回调，不改变概率、排序、输出或随机状态。
保持 Math Attention 和 FP32；没有开启 AMP、TF32 调优或 Flash Attention。

可在同一机器、同一权重和同一 epoch 下补齐不重叠的剩余目标：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_formula_beam_cuda_v04.ps1 -Start 32 -Count 224
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_formula_beam_cuda_v04.ps1 -Start 256 -Count 256
```

每次启动检查新实验停止锁、CPU gate 和两份 checkpoint 哈希。
发现故障、子进程失败、监控异常或超时会写入停止锁，停止后续作业，禁止自动重试。
本机新授权的 NVIDIA 工作不清除或复用 AMD 停止锁。

## Windows 监控边界

监督器在作业前、作业中、退出后检查 System 4101 和 Application WER LiveKernelEvent 141/117，
并记录 NVIDIA 驱动、温度、显存、利用率及功耗。
持续 epoch 包括作业间隙，事件用显式 UTC 比较，避免 JSON 自动转换日期带来的误判。

当前普通权限不能读取 `C:\Windows\LiveKernelReports\WATCHDOG`。
该目录不会被冒充为“已扫描且无 dump”：报告明确保存无法读取的目录，
使用可读取的事件日志监测 GPU timeout，并继续扫描可访问的 GPU dump 目录。
新出现的访问权限变化视为监控异常；不能仅凭事件计数为零断言没有发生任何 GPU 故障。

进程 RAM 改用 Windows `GetProcessMemoryInfo` 的 PeakWorkingSetSize；Linux 保留 `resource` 实现。
两者统一以 KiB 报告。每个作业结束再次计算两份 checkpoint 的 SHA256。

## 结果

```powershell
python scripts/report_formula_beam_cuda_v04.py
```

报告：[`POLAR_FORMULA_BEAM_CUDA_V04_RESULTS.md`](../results/POLAR_FORMULA_BEAM_CUDA_V04_RESULTS.md)，
以及同名 JSON。聚合验证每条样本的冻结 cohort 索引、原始行和作业 contract，拒绝重复目标及跨设备混合。
只有同一 case 覆盖全部 512 个目标才标记 `complete_512`；子集明确标记 `bounded_subset`。
失败作业不纳入指标，原始输出和严格过滤后的输出分别保留。
旧 AMD 的 241 条中断前缀不会拼成本机的 512 结果。

公式正确但目标图错误的候选仍报告为错误结构；同式异构体允许保留，按 Decoder 分数排序。
不加入数据库白名单，不改公式，不强行修复失败结构。
私有结构、原始候选、权重和数据均留在 G 盘，仅发布聚合结果。

## 首次接续发现的验证异常与修订版

首次 native 运行完成 38 个正式分块及 2 个预检后，Morgan Beam 50 的元素＋语法分块在
第 79 个目标的候选验证阶段退出。已完成的 78 条中断前缀仅保留诊断，不进入主比较。
停止原因为 `child_failed`；没有 CUDA 错误、System 4101 或 WER GPU timeout 事件，权重未变。
同一目标在禁用 GPU 的 CPU 路径复现同一 `KekulizeException`。

原始候选可通过第一轮解析，但 `canonical_polar` 返回的文本不能再次通过 RDKit sanitize。
修复在原有 Polar 验证路径增加 canonical 文本重新解析，失败则拒绝该候选。
原始 `rdkit_valid` 仍保留原文的解析结果；`polar_valid` 还要求 canonical 往返检查通过。
没有修复 SMILES、放宽价态、调整公式或修改 Builder。
同时增加解码后、化学检查前的 `raw_beams_private.jsonl`，便于后续验证异常时保留原始输出。

修订版 CPU gate 为 364 项测试通过，两份权重和 512 目标审计再次通过。
故障目标的 CPU 重放正常结束；GPU 修订运行也已越过原故障目标。
复用旧 native 完整分块前，用修复后的 `polar_checks` 重审全部 69,190 个候选（含预检），
有效性和 canonical graph 判定零变化；Decoder、约束及权重哈希相同。
审计通过后才允许报告读取这些完整结果，并核对每个源文件的 SHA256。

修订版工作目录为 `outputs/formula_constrained_v04_rtx4070_20261020_r1`。
原 native 目录的停止锁、错误日志及 7 个执行源码快照保持原样。
新目录有独立 CPU gate，但监控 epoch 继承原 UTC 起点，持续覆盖修复间隙。
这次是已定位、已在 CPU 上修复验证后的新版本运行，不是对未定位 GPU 故障的盲目重试。

```powershell
python scripts/check_formula_beam_v04.py --gpu-backend native_cuda `
  --experiment-root outputs/formula_constrained_v04_rtx4070_20261020_r1
python scripts/audit_completed_formula_cuda_v04.py `
  --prior-root outputs/formula_constrained_v04_rtx4070_20261020 `
  --root outputs/formula_constrained_v04_rtx4070_20261020_r1
python scripts/report_formula_beam_cuda_v04.py
```

上面的前两个命令用于建立全新修订目录；本机已有的 gate 和审计受保护，不能覆盖重跑。
修订 GPU 作业必须显式传 `--experiment-root ..._r1`；PowerShell 批量启动器传 `-ExperimentRoot ..._r1`。
报告默认读取修订目录及其经审计的旧 native 完整分块，仍不读取 AMD 中断候选。
