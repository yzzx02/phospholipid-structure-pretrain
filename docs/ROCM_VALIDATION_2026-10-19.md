# WSL 迁移与 GPU 验证记录：2026-10-19

## 当前结论

**已找到并验证 Math attention 绕行方案。FP/GINE 均完成 32 条真实结构过拟合、20,000 目标池 / 1,000 steps GPU smoke，并各实际恢复至第 1,001 步；这条路径的所有受监控运行均未出现新增 Windows GPU 超时转储。训练进程现已结束。**

默认 attention 路径曾触发黑屏，最初的短算子成功没有保证持续稳定。后续固定 Math SDPA 并关闭进程内 MHA fastpath 后完成下述验收；原 fused attention 路径是疑似触发点，尚未独立证明是唯一根因。本轮结果不证明长时间正式训练已稳定，也不能据此判断显卡硬件有无损坏。

## Linux 位置与环境

- 代码、数据和训练输出：`${PROJECT_ROOT}/`，WSL ext4。
- 归档：`${DATA_ARCHIVE_ROOT}/phospholipid_v03_training_data.tar.gz`。
- Python：`${HOME}/miniconda3/envs/torch-rocm721/bin/python`，3.12.13。
- Torch：`2.9.1+rocm7.2.1.gitff65f5bc`；HIP `7.2.53211-e1a6bc5663`。
- GPU 识别为 AMD Radeon RX 9070 XT，报告显存 16,974,905,344 bytes。
- 需要 `HSA_ENABLE_DXG_DETECTION=1`；本项目 `scripts/run_rocm.sh` 提供该设置和固定解释器。
- ROCDXG `1.2.0`，系统 ROCm `7.2.1.70201`，WSL kernel `6.18.33.1-microsoft-standard-WSL2`。
- 保留了现有 Torch、torchvision、torchaudio、Triton、NumPy 和 ROCm 版本。新增项目依赖前以 constraints 和 dry-run 检查，安装后 `pip check` 通过。

## 通过的检查

1. 下载归档 SHA-256：`698f652c2b992536637a8ddf1e606383b1bab394d34d5845dd4bd9fdccd2f7f7`，与交付校验文件匹配。
2. 全部 320 个交付数据文件的大小和 SHA-256 匹配，数据总量 4,334,776,708 bytes。
3. 正式库 Parquet metadata 确认 2,015,000 行；train / validation / test 为 1,813,451 / 100,792 / 100,757。
4. `verify_pretraining_v03.py` 检查源库、冻结索引、派生数组、shards、train bank、检索和公式统计，277 项派生检查通过。黑屏并停止训练后再次运行仍通过。
5. `check_rocm.py` 的 FP/GINE × FP32/BF16/FP16 共 6 组前后向、有限梯度、参数更新、checkpoint 重载和继续更新通过。
6. 最终本机 CPU 回归 **276 项通过**（8.63 秒）；Ruff lint 和 format check 通过。覆盖新增的训练集选样、自由生成验收门槛、冻结文件保留、attention 状态恢复/模型输出等价、checkpoint 权重严格核验和输出容差检查。

日志位于 `outputs/pretraining_v03/rocm_setup/` 和 `outputs/pretraining_v03/rocm_preflight/`。

## 首次默认 attention 路径的过拟合与故障

新增 `scripts/overfit_pretraining_v03.py`。使用 32 个不同 TRAIN 头基的真实结构，包含训练集最短和最长极性片段；模板固定为跨头基 train-only 检索，测试诊断单独关闭 dropout，不改正式 smoke 配置。验证集和测试集均未用于训练。

预先设定门槛：从 BOS 自由 greedy 生成的 EOS rate ≥95%、Polar 合法率 ≥90%、目标 Polar 规范图重建率 ≥80%。不强制 EOS，不修复 SMILES。该诊断检查记忆能力，不代表泛化或 MS/MS 鉴定准确率。

FP 运行最后一条正常训练日志为 step 40，loss 从 step 1 的 10.1938 降到 1.2270。首次训练后生成评价计划在 step 200，因此尚无学会 EOS 或合法 Polar 的验收结果，也尚未保存 step 200 checkpoint。

用户报告多次黑屏后立即停止该进程，确认已退出；没有启动新的 GPU 负载。
用户随后确认停止训练后屏幕恢复稳定。

Windows 故障记录：

- `LiveKernelEvent 141`，以及 AMD watchdog / AMD report 事件。
- `WATCHDOG-20261019-1817.dmp`、`WATCHDOG-20261019-1818.dmp`、`WATCHDOG-20261019-1819.dmp` 对应本次黑屏期间。
- 当前 Windows 显卡驱动 `32.0.31041.1004`，驱动日期 2026-08-17。
- 相关事件摘录保存于 `outputs/pretraining_v03/rocm_setup/windows_gpu_faults.json`。

微软将 0x141 定义为显示引擎未及时响应的 live dump；这个事件本身不能判断硬件是否损坏。[微软说明](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x141---video-engine-timeout-detected)

## Math attention 绕行与复测

`utils/attention.py` 使用 PyTorch 的 `sdpa_kernel(SDPBackend.MATH)`，并在作用域内关闭 MHA fastpath，退出后恢复原有设置。没有替换核心依赖或改变模型的网络结构。两种编码器的 CPU 完整模型输出对照通过。[PyTorch 2.9 API](https://docs.pytorch.org/docs/2.9/generated/torch.nn.attention.sdpa_kernel.html)

先对 FP/GINE 分别运行 8 步、batch 2 的真实最短/最长标签（12/84 tokens）同步诊断，再分别执行过拟合和预设 smoke。Windows 主机监控 WATCHDOG、AMD_WATCHDOG、AMD_REPORT_UM 目录，出现新转储或进度停滞则停止指定训练进程。所有这些 Math 路径运行均无新增故障转储。

### 32 条真实 TRAIN 结构过拟合

| 指标 | FP | GINE |
|---|---:|---:|
| 实际 optimizer steps | 200 | 200 |
| EOS rate | 100% | 100% |
| 合法 Polar | 100% | 93.75% |
| 目标 Polar 规范图重建 | 100% | 93.75% |
| 链集合 exact | 100% | 100% |
| 主机监控总时长 | 27.09 秒 | 27.52 秒 |

两者均达到预先定义的门槛。这里评价的是训练样本记忆能力，不能作为未见结构准确率。

### 20,000 目标池 GPU smoke 与实际续训

保留预设配置：FP batch 16；GINE batch 8 × gradient accumulation 2；FP32；workers 2；普通/跨头基检索混合。分别训练 1,000 步，并在新进程中从相应 checkpoint 恢复至 1,001 步。

| 指标 | FP | GINE |
|---|---:|---:|
| 累计 optimizer steps | 1,001 | 1,001 |
| 真实 train 目标池 | 20,000 | 20,000 |
| 实际消费 / 不同 train 目标 | 16,016 | 16,016 |
| Teacher forcing validation 目标 | 2,000 | 2,000 |
| Validation polar token accuracy | 69.85% | 68.91% |
| 自由生成 validation 目标 | 128 | 128 |
| 自由生成 EOS | 96.09% | 96.09% |
| 自由生成合法 Polar / 组装成功 | 26.56% | 20.31% |
| 自由生成 polar exact / 完整图 exact | 0% / 0% | 0% / 0% |
| 续训进程峰值 allocated VRAM | 0.692 GiB | 0.660 GiB |
| 参数更新及 checkpoint 核验 | 通过 | 通过 |

20,000 是抽取的目标池大小，不是已经全部训练遍历。没有训练全部 201.5 万结构，没有用 test 调参或训练，没有执行 MS/MS 微调。128 条自由生成不强制 EOS、不修复或后验筛式。上述质量仍是初步工程 smoke，尚不足以用于实际结构鉴定。

保留的中间问题和修正：

- FP 首次 1,000 步完成后，加载 195,306 个头基规则的 CPU 阶段被 20 秒无日志监控误停。已改为分批读取和报告进度；从其 1,000 步 checkpoint 续训并完整评估，没有重训或隐去中断记录。
- GINE 首次 1,000 步完成后，GPU 输出比较因 2 / 28,224 个元素超过旧绝对容差而停止，所报最大绝对误差约 `1.28e-6`。改为所有权重/buffer 严格零误差校验、CPU 输出严格相等、GPU 输出采用 PyTorch 的 dtype 默认容差。FP32 官方默认是 `rtol=1.3e-6, atol=1e-5`；新单测仍拒绝权重变化和超容差输出。[官方容差](https://docs.pytorch.org/docs/2.9/testing.html)
- GINE 实际续训后的权重/buffer 严格一致，chain/polar logits 最大绝对差分别为 `2.86e-6 / 3.34e-6`，在 FP32 默认容差内。FP 续训的旧核验路径已通过更小的 `atol=1e-6`，且 logits bitwise identical；其报告不冒充新增加的独立 state_dict 审计。
- GPU RNG 状态在加载到设备后先转回 CPU ByteTensor 再恢复，实际 FP/GINE 续训均通过。

吞吐和显存 JSON 属于各自**续训进程（1 步 + 验证/生成）**，不能当作首次完整 1,000 步的吞吐或峰值。两次首次运行均已保存 1,000 步 checkpoint，但在各自报告阶段结束，未产出完整的首次资源报告。

### 产物与复现

```text
outputs/pretraining_v03/rocm_overfit/fingerprint_math/
outputs/pretraining_v03/rocm_overfit/graph_math/
outputs/pretraining_v03/rocm_smoke/fingerprint_resume/
outputs/pretraining_v03/rocm_smoke/graph_resume/
outputs/pretraining_v03/rocm_setup/validation_summary.json
```

`outputs/pretraining_v03/rocm_smoke/validated/{fingerprint,graph}` 是到已完成续训目录的 Linux 链接，供后续完整训练的 smoke gate 读取。保留原始 `fingerprint/`、`graph/` 和故障/误停日志。

本机后续运行需显式选择 Math 路径，例如在 WSL 仓库内：

```bash
./scripts/run_rocm.sh train_structure_real.py --device rocm --attention-backend math --config configs/pretraining_v03_rocm_fp.yaml --steps 1002 --resume outputs/pretraining_v03/rocm_smoke/fingerprint_resume/checkpoint/last.pt --output outputs/pretraining_v03/rocm_smoke/fingerprint_resume
```

上面是复现后续单步续训的示例，本次已在 1,001 步结束。完整多轮预训练尚未运行，后续如启动应同样使用已验证的 Math 路径，按实际生成质量决定实验方案。当前 Math 路径的真实训练验证为 FP32；初始 6 组 AMP 短算子结果来自默认路径，不冒充 Math 路径 AMP 的重新验收。

未修改 TDR 注册表、Windows 显卡驱动、系统 ROCm/ROCDXG 或 Torch/NumPy/Triton 等核心版本；`pip check` 通过。320 个交付文件的大小/哈希复核保持一致。若后续需要替换核心环境，应按父目录 `AGENTS.md` 的依赖规则先确认。

本次另修正 `train_structure_real.py`：smoke 抽样名单保存到运行目录的 `sample_rows/`，避免覆盖迁移包内原有的名单及其哈希。冻结库、split 和检索缓存保持原样。
