# RX 9070 XT：WSL Ubuntu + ROCm

**GPU执行状态：待运行。** 当前电脑只执行CPU真实数据smoke；下面是另一台设备的明确步骤。推荐WSL2 Ubuntu24.04、Python3.12。Windows驱动、WSL运行时和HIP PyTorch按[AMD WSL支持矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)及[AMD安装指导](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)选择相互匹配的版本。

## 1. 保留已工作的AMD环境

在准备实际训练的Python环境中先检查：

```bash
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else "no GPU")'
```

`torch.version.hip`必须非空、`torch.cuda.is_available()`必须True。ROCm使用`torch.cuda`接口是正常情况。没有HIP构建时，先按AMD官方对应WSL/显卡版本安装；不要运行本仓库CPU wheel命令，也不要用CPU约束文件强行替换已工作的AMD Torch。

克隆代码，复制私有数据到WSL Linux文件系统中的SSD目录：

```bash
gh repo clone yzzx02/phospholipid-structure-pretrain
cd phospholipid-structure-pretrain
# 当前环境已安装匹配的AMD Torch；以下命令不安装或升级Torch。
python -m pip install --no-deps -e .
python -m pip install --no-deps torch-geometric==2.8.0.post1
python -m pip install 'rdkit==2026.03.3' 'pyarrow>=20,<25' 'usearch==2.26.4' 'omegaconf>=2.3,<3' PyYAML psutil aiohttp fsspec jinja2 pyparsing requests tqdm xxhash
python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.__version__, torch.version.hip)'
```

PyG基础GINE依赖PyTorch的scatter/index_add；不要求安装NVIDIA CUDA wheels或额外torch-scatter/torch-sparse/pyg-lib。NumPy沿用当前AMD环境的兼容版本，以上命令不主动升级它。若要运行dummy/Lightning入口，再单独补齐已验证的Lightning依赖；真实入口不使用Lightning。

需要迁移的目录为 `data/structure_library_v0.3/`、`data/private/morgan_index_v0.3/`、`data/private/pretraining_v03/`。后者包括合法train index、两模式缓存、词表、公式统计和labels/split manifests。路径保持相对仓库可迁移，校验入口会检查哈希。不要复制v0.2的全库检索缓存来替换训练缓存。

## 2. 算子、AMP和checkpoint预检

```bash
python scripts/verify_pretraining_v03.py
python scripts/check_rocm.py --output outputs/pretraining_v03/rocm_preflight
```

预检检查HIP/GPU、GINE和FP两条完整模型的forward/backward、Transformer attention、FP32/BF16/FP16、参数更新、checkpoint保存及继续更新。预检采用dummy fixtures核验硬件算子，不能替代下一步真实数据smoke。失败时保留错误，不能假装通过或自动回退CPU。

## 3. 真实GPU smoke，先FP后GINE

```bash
python train_structure_real.py --device rocm --config configs/pretraining_v03_rocm_fp.yaml --output outputs/pretraining_v03/rocm_smoke/fingerprint
python train_structure_real.py --device rocm --config configs/pretraining_v03_rocm_gine.yaml --output outputs/pretraining_v03/rocm_smoke/graph
```

初始20,000 train /2,000 validation /1,000 optimizer steps，均从冻结合法groups抽取，test不进入训练/调参。FP batch16、GINE batch8×accum2、workers2、FP32是**待实测起点**；确认显存/吞吐后可在独立配置中测试BF16/FP16、batch、accum、workers和prefetch。允许20k–50k/2k–5k/1000–5000 steps，不修改任务或冻结数据。

续训示例（steps为累计数）：

```bash
python train_structure_real.py --device rocm --config configs/pretraining_v03_rocm_fp.yaml --steps 1001 --resume outputs/pretraining_v03/rocm_smoke/fingerprint/checkpoint/last.pt --output outputs/pretraining_v03/rocm_smoke/fingerprint
```

每个输出目录保存resolved config、训练/teacher-forcing指标、独立自由生成指标、steps/s、samples/s、加载时间、RAM和VRAM，以及私有checkpoint。需实际完成一次GPU恢复并检查日志；本机的CPU恢复不能证明HIP恢复已验证。

## 4. 验收后独立启动full

以下命令**本轮不执行**。完整数据实验仅用train组1,813,451条训练，100,792条validation，test保持封存。120,000 steps为配置占位实验长度，后续依据GPU实测确定；当前不是已验证的最优训练量。

```bash
python train_structure_real.py --device rocm --config configs/pretraining_v03_full_fp.yaml --allow-full-training --gpu-smoke-root outputs/pretraining_v03/rocm_smoke --output outputs/pretraining_v03/full/fingerprint
# GINE为单独实验：
python train_structure_real.py --device rocm --config configs/pretraining_v03_full_gine.yaml --allow-full-training --gpu-smoke-root outputs/pretraining_v03/rocm_smoke --output outputs/pretraining_v03/full/graph
```

Full入口要求显式full配置、`--allow-full-training`和两种encoder已完成真实GPU smoke报告。Smoke不会调用full；没有任何自动训练串联。Full默认每500 steps原子替换checkpoint，恢复时检查所有数据和模型契约。

当前电脑没有RX9070XT，因此HIP可用性、GINE/Transformer GPU backward、AMP数值稳定性、GPU续训、显存和GPU吞吐均为**待运行**，GitHub CPU CI也不能替代它们。
