# RX 9070 XT：WSL Ubuntu + ROCm

当前本机只完成 CPU 程序验证。以下提供另一台 9070 XT 机器的安装顺序和运行入口，尚未声称 GPU 实测成功。

## 环境选择

建议 WSL2 + Ubuntu 24.04 + Python 3.12。AMD 发布的 Radeon WSL 支持矩阵明确列出 RX 9070 XT，以及 PyTorch 2.9.1 + ROCm 7.2.1 的官方支持组合；该组合是本文的参考基线，不代表最新版本。驱动和运行时请按实际选择的版本成套安装。

- [AMD WSL 支持矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
- [AMD Radeon WSL 安装指导](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)
- [AMD PyTorch release wheels](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
- [AMD ROCm 当前安装文档](https://rocm.docs.amd.com/en/latest/install/rocm.html)

先安装 Windows 端兼容的 AMD 驱动，再安装 WSL/Ubuntu 对应的 ROCm 用户态运行时。WSL 的 GPU 暴露依赖 Windows 驱动和 `/dev/dxg`，不要照搬原生 Linux 的内核驱动安装或 NVIDIA CUDA 安装命令。

AMD 的 WSL 安装路线已经出现 ROCDXG 迁移。若选择需要 ROCDXG 的旧运行时，遵循官方安装说明；[AMD ROCDXG 说明](https://github.com/ROCm/librocdxg) 指出 legacy runtime 需要 `HSA_ENABLE_DXG_DETECTION=1`，从 ROCk 7.13 起去掉该要求，其源码已迁至 rocm-systems。本项目不自动修改显卡驱动、runtime library 或全局环境变量。

## 获取项目和创建环境

私有仓库需要你在新机器上先用 `gh auth login` 或配置 GitHub SSH key。把代码放在 WSL 自己的 Linux 文件系统中，例如 `~/projects/`，方便管理 Python 环境和缓存。

```bash
mkdir -p ~/projects
cd ~/projects
gh repo clone yzzx02/phospholipid-structure-pretrain
cd phospholipid-structure-pretrain
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

如果已装好 ROCm PyTorch，先运行下面的检查，无需重复安装：

```bash
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
```

注意 venv 通常隔离系统包，因此应在**当前 venv** 内安装 HIP PyTorch。

## 安装参考基线的 PyTorch

以下 URL 取自 AMD 发布的 Linux Python 3.12 release wheels 页面，使用前需要已完成 WSL GPU runtime 安装。项目只需 torch，不使用 torchvision/torchaudio。ROCm 7.2.1 的匹配 Triton wheel 一起安装；模型本身不使用 `torch.compile`。

```bash
python -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'
python -m pip install -c constraints-rocm.txt -e '.[dev]'
python scripts/check_rocm.py
```

项目安装约束会固定已测试的核心包版本；`torch==2.9.1` 接受 AMD wheel 的本地版本后缀，不会要求替换成 CPU wheel。ROCm 约束保守使用 NumPy 1.26.4，CPU 已测试 NumPy 2.2.3。这里不是整个系统和 Python 依赖的完整 lockfile。

PyG 从 2.3 起基础使用不需要额外扩展，见 [PyG 官方安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。本模型只用 GPSConv、GINE、dense attention 和 sum pooling，因此先安装 `torch-geometric` 即可，不额外装 NVIDIA CUDA wheels、`torch-scatter`、`torch-sparse` 或 `pyg-lib`。

## 验证与训练

`check_rocm.py` 会检查 `torch.version.hip`、GPU 可用性、设备名称，并实际执行 GPU attention、index_add 及 backward。接着用整个模型核验 graph 和 FP 路径：

```bash
python -m pytest -q
python train_structure_dummy.py --config configs/structure_pretrain_graph.yaml --device rocm --steps 2 --output outputs/rocm_graph
python train_structure_dummy.py --config configs/structure_pretrain_fp.yaml --device rocm --steps 2 --output outputs/rocm_fp
```

ROCm 仍使用 PyTorch 的 `torch.cuda` API，Lightning 的 accelerator 仍为 `gpu`，不需要把设备字符串改为 `hip`。`--device rocm` 在环境不满足条件时退出，不会默默回退 CPU。

两条 FP32 路径都通过后，再试：

```bash
python train_structure_dummy.py --config configs/structure_pretrain_graph.yaml --device rocm --precision 16-mixed --steps 2 --output outputs/rocm_mixed
```

当前默认 batch size=2 用于验证。真实训练接入后再按最大原子数、Top-K、headgroup 长度和 GPU 实测显存调 batch；单凭约 16M 参数不能承诺最大 batch size，GPS 的分子内部 attention 还取决于原子数。

## 报错定位

- `hip=None`：当前 venv 的 torch 不是 ROCm 构建，或激活了错误的环境。
- `gpu_available=False`：先检查 Windows 驱动、`/dev/dxg` 和对应 WSL runtime，按官方路线排查。
- `device kernel image is invalid`：核对 torch wheel、ROCm 和显卡架构支持，使用同一套兼容版本；不默认加架构伪装环境变量。
- NumPy ABI 或依赖冲突：确认按 `constraints-rocm.txt` 安装，保留错误输出再排查。
- OOM：先维持 batch=2/FP32 跑通，再尝试 mixed precision 或减 batch；不改变已固定的模型宽度。

本仓库的 GitHub Actions 只验证 Linux CPU，不能替代真实 9070 XT 验证。
