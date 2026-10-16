# STRUCTURE-ONLY PRETRAINING v0.1

**当前阶段：独立结构库构建，不训练模型。** 已完成 phosphate 极性片段扩充，得到 195,305 个候选头基，并完成最终库上限 **200 万**的 family-aware 预算模拟。详见 [扩充方法](docs/PHOSPHATE_EXPANSION.md)和[扩充与预算结果](docs/PHOSPHATE_EXPANSION_RESULTS.md)。当前仍停在 dry-run，正式组装前需补足 P 链覆盖。基础入口见[建库执行与验证说明](docs/LIBRARY_BUILDER.md)及[找到的 MS-DIAL 原始库](docs/MSDIAL_LOCAL_SOURCES.md)。现有预训练网络、retrieval、loss 和训练脚本保持原实现。

当前仅实现**基于结构检索模板的磷脂结构预训练框架**。只保留 Morgan fingerprint 基线和三层 GINE 图模型，后续网络和训练目标共享。真实磷脂结构库、中性分子式库、headgroup 切割规则和 chain vocabulary 由后续数据构建阶段提供。

本仓库的示例使用小型 alkyl phosphate toy structures 和**人为输出标签**，不是化学上配套的真实磷脂训练集。它们只核验接口、张量、梯度、训练和 checkpoint。当前不提供预测准确率结论。

```text
Target phospholipid structure
        ↓
Morgan structural retrieval
radius=2 / 2048 bits / Tanimoto
max Top-10 / min similarity 0.4
exclude target itself and equivalent/duplicate structures
        ↓
K template full structures
        ↓
┌──────────────────────────────────────────┐
│ Structure Encoder                        │
│ Baseline: Morgan FP → MLP (2048→512→256) │
│ Graph: 3-layer GINE                       │
│ hidden=128 / layers=3                     │
│ sum pooling → MLP → 256D                │
└──────────────────────────────────────────┘
        ↓
structure embedding 256D
+ F_template 10D + ΔF 10D
+ structure_similarity 1D + formula_available 1D
        ↓
concat 278D → MLP (278→512→512)
        ↓
Template tokens [B, K, 512]
        ↓
Template Set Transformer + learned [TCLS]
2 layers / 8 heads / FFN 2048 / no positional encoding
template dropout=0.2; at least one valid template
        ↓
Architecture       4 Chain Slots       Headgroup SMILES
2-class logits    one token per slot  causal Transformer Decoder
                                     cross-attention to template memory
```

输入模板为完整结构；不在输入端分解架构、链或头基。输出 chain slots 采用固定 canonical sort：O → P → A，同类按 C 数、双键数升序，NULL 补足 4 个；这只是确定性顺序，不代表 sn 位置。当前不预测双键位置、E/Z 或复杂立体化学；图仍保留输入 SMILES 的二维连接关系，不重新指定双键位置。

分子式元素顺序固定为 `[C,H,N,O,P,S,F,Cl,Br,I]`，必须由调用方提供**中性分子式**。先在原始整数计数上计算 `ΔF=F_query−F_template`，保留负值，再分别用训练检索对的均值/标准差标准化。常数维度使用 std=1。query formula 缺失时 raw delta 和标准化后的 delta 都为 0，`formula_available=0`，模板自身分子式保留。统计值保存为 checkpoint buffers，验证集不重新拟合。没有额外 query formula token、相似度 encoder、rank/source embedding。

## 安装与 CPU 快速验证

Python 3.12 推荐用于 WSL；本地 CPU 已在 Python 3.13 验证。先安装所需 PyTorch 构建，再安装项目，避免覆盖已有 ROCm PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 以下 torch 安装命令专供 CPU 环境。
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-tested.txt -e '.[dev]'
python -m pytest -q
python train_structure_dummy.py --config configs/structure_pretrain_graph.yaml --device cpu --steps 2 --output outputs/graph
python train_structure_dummy.py --config configs/structure_pretrain_fp.yaml --device cpu --steps 2 --output outputs/fingerprint
python scripts/count_parameters.py
python scripts/check_encoder_batch.py
```

Windows venv 激活命令为 `.venv\Scripts\Activate.ps1`。训练入口默认 CPU、batch size=2、32-bit、AdamW lr=1e-4、weight decay=0.01、clip=1。两种 YAML 都使用完整默认模型，没有偷偷缩小测试模型。CLI 只接受 structure pretraining。

每次运行保存 `last.ckpt`、`run_summary.json` 和 CSV loss/metric 日志。继续训练例：

```bash
python train_structure_dummy.py --config configs/structure_pretrain_graph.yaml --steps 3 --resume outputs/graph/last.ckpt --output outputs/resume
```

`--steps` 指累计 optimizer steps。恢复 model、optimizer 和 Trainer；dummy DataLoader 不保存 epoch 中途游标，因此不承诺完全相同的 batch replay。此次改造前的图模型 checkpoint 不兼容，应重新训练；新的 GINE checkpoint 可正常续训。

## RX 9070 XT：WSL Ubuntu + ROCm

安装顺序和 GPU 检查见 [WSL_ROCm.md](docs/WSL_ROCm.md)。本项目使用 PyTorch/PyG 自带基础算子，未引入 CUDA 专用 FlashAttention 或额外 PyG C++ 扩展。**已验证 CPU；尚未在你的 9070 XT 上实测。**

在正确安装 HIP PyTorch 的 WSL 环境中：

```bash
python scripts/check_rocm.py
python train_structure_dummy.py --config configs/structure_pretrain_graph.yaml --device rocm --steps 2 --output outputs/rocm_graph
python train_structure_dummy.py --config configs/structure_pretrain_fp.yaml --device rocm --steps 2 --output outputs/rocm_fp
# FP32 成功后再尝试混合精度。
python train_structure_dummy.py --config configs/structure_pretrain_graph.yaml --device rocm --precision 16-mixed --steps 2 --output outputs/rocm_fp16
```

## 完整目录与模块职责

```text
phospholipid-structure-pretrain/
├── .github/workflows/cpu.yml            # Linux CPU 自动验证
├── .gitattributes
├── .gitignore
├── README.md
├── pyproject.toml
├── constraints-tested.txt
├── constraints-rocm.txt
├── train_structure_dummy.py             # CPU/ROCm train、validate、resume
├── configs/
│   ├── structure_pretrain_graph.yaml
│   └── structure_pretrain_fp.yaml
├── docs/
│   ├── DATA_SCHEMA.md                   # 后续数据输入契约、TODO
│   ├── WSL_ROCm.md                      # 9070 XT 安装与运行说明
│   ├── VERIFICATION.md                  # 已执行验证与边界
│   ├── ENCODER_SIMPLIFICATION.md        # GINE 改造、特征编码和小 batch 报告
│   ├── encoder_batch_shapes.json        # B=2、M=5 的实际 shape / backward 记录
│   └── verification_summary.json        # 精简的实际运行记录
├── scripts/
│   ├── check_rocm.py                    # HIP attention/scatter 梯度核验
│   ├── check_encoder_batch.py           # 3+2 模板 batching/padding/梯度检查
│   └── count_parameters.py              # 实例化模型，去重统计参数
├── src/structure_pretrain/
│   ├── __init__.py
│   ├── config.py                       # 有类型配置和维度校验
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py                   # StructureRecord/TargetRecord
│   │   ├── structure_dataset.py        # 外部标签与检索结果转换
│   │   ├── dummy_structure_dataset.py  # 小型合成测试数据
│   │   ├── batching.py                 # K≤10 padding，真实模板展平
│   │   ├── chain_vocab.py              # 外部词表、canonical sort、NULL
│   │   ├── smiles_tokenizer.py         # BOS/EOS/PAD，attachment 原子词元
│   │   └── molecular_graph.py          # 7 原子特征、3 键特征、FP
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── structural_retrieval.py     # 可替换的 Protocol、Morgan/Tanimoto
│   ├── models/
│   │   ├── __init__.py
│   │   ├── graph_encoder.py            # 三层 GINE、sum pooling
│   │   ├── fingerprint_encoder.py      # 2048→512→256 baseline
│   │   ├── template_token_projector.py # 278→512→512，两条路径共享
│   │   ├── template_set_transformer.py # TCLS、mask、template dropout
│   │   ├── architecture_head.py        # 512→256→2
│   │   ├── chain_head.py               # 512→256→4×V_chain
│   │   ├── headgroup_decoder.py        # 2 层/8 heads/FFN2048，teacher forcing
│   │   └── structure_pretrain_model.py # 模型组合与统一输出
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lightning_module.py         # device transfer、优化器、checkpoint
│   │   └── losses.py                   # 三项 CE，PAD 忽略及基础指标
│   └── utils/
│       ├── __init__.py
│       ├── formula.py                  # 10 元素检查、train-only 标准化
│       ├── masks.py                    # 模板 dropout 至少保留一个
│       └── seed.py                     # 随机种子、CPU 线程数
└── tests/
    ├── conftest.py
    ├── test_graph_encoder.py
    ├── test_graph_batch_padding.py
    ├── test_fingerprint_encoder.py
    ├── test_template_projector.py
    ├── test_template_transformer.py
    ├── test_structure_pretrain_forward.py
    ├── test_structure_pretrain_backward.py
    └── test_retrieval_and_schema.py
```

`outputs/` 和本地虚拟环境不上传 GitHub。

## 关键 tensor shape

`B` 为 batch size，`K=10` 为补齐容量，`M` 为当前 batch 的真实模板总数，`L` 包含 BOS/EOS 的补齐头基长度。

| 张量 | Shape |
|---|---|
| FP / 图节点 x / 图键 edge_attr | `[M,2048]` / `[N_atoms,7]` / `[N_edges,3]` |
| 原子 / 键 embedding | `[N_atoms,128]` / `[N_edges,128]` |
| 图 sum pooling | `[M,128]` |
| 真实结构 embedding | `[M,256]` |
| 补齐结构 embedding | `[B,K,256]` |
| template formula / signed delta | `[B,K,10]` / `[B,K,10]` |
| similarity / formula available / template mask | 各 `[B,K]` |
| 拼接输入 / template tokens | `[B,K,278]` / `[B,K,512]` |
| template memory / global / memory padding mask | `[B,K+1,512]` / `[B,512]` / `[B,K+1]` |
| architecture logits | `[B,2]` |
| chain logits | `[B,4,V_chain]` |
| headgroup input / label | 都为 `[B,L−1]`，分别右移一位 |
| headgroup logits | `[B,L−1,V_smiles]` |

数据 `template_mask=True` 表示有效；Transformer `padding_mask=True` 表示忽略。TCLS 总是有效，padding/dropped 模板在 encoder 和 decoder attention 中都屏蔽。

## 参数与验证

下列为实际实例化默认配置的可训练参数，dummy `V_chain=6`、`V_smiles=47`。真实词表改变后需重新统计。

| 模块 | Graph | Fingerprint |
|---|---:|---:|
| Structure encoder | 154,755 | 1,181,952 |
| TemplateTokenProjector | 406,528 | 406,528 |
| Template Set Transformer | 6,306,304 | 6,306,304 |
| Architecture head | 131,842 | 131,842 |
| Chain head | 137,496 | 137,496 |
| Headgroup decoder | 8,490,031 | 8,490,031 |
| **总参数** | **15,626,956** | **16,654,153** |

损失 `L = CE_arch + CE_chain + CE_headgroup`，三项权重均为 1。NULL 是有监督的合法 chain 类；headgroup CE 忽略 PAD、保留 EOS。没有额外辅助损失、Hungarian matching 或 formula hard constraint。

验证记录见 [VERIFICATION.md](docs/VERIFICATION.md)。

图编码器改造报告见 [ENCODER_SIMPLIFICATION.md](docs/ENCODER_SIMPLIFICATION.md)。实验只比较 Morgan 和 GINE；数据划分、随机种子、下游网络、优化器、学习率、loss 和训练步设置保持一致。

## MS-DIAL 真实链证据提取

独立提取模块和运行说明见 [MSDIAL_CHAIN_EXTRACTION.md](docs/MSDIAL_CHAIN_EXTRACTION.md)。
两版全量扫描结果及审核要点见 [MSDIAL_CHAIN_EXTRACTION_RESULTS.md](docs/MSDIAL_CHAIN_EXTRACTION_RESULTS.md)。
从本地 2025/2022 LBM2 读取元数据、以分子图提取 A/O/P 链、统计双键位置版本并输出人工审核候选。
原提取阶段只输出证据。后续已按 2025 单位置版本规则冻结 322 条链，位置保持不变；
本地大表在 `data/private/msdial_chain_review/`，不上传源库。

## PubChem 极性片段扩展

独立 CPU 化学流程见 [PUBCHEM_POLAR.md](docs/PUBCHEM_POLAR.md)。
`scripts/harvest_pubchem_polar.py` 先验证正负对照，再依次执行 1k/query、10k/query 和 full。
检索 phosphate / phosphonate / polyphosphate，保留真实 PubChem CID 和所有酸性连接位点，
输出标准化去重后的人工审核表；本阶段不生成批准版头基表、不训练、不生成完整脂质库。
审核包在 `data/private/pubchem_polar_review/`；原始大表与 SSD 缓存不上传 GitHub。
分阶段实测数量、完整性核验和 20 个真实来源片段见
[PUBCHEM_POLAR_RESULTS.md](docs/PUBCHEM_POLAR_RESULTS.md)。

## 后续数据 TODO

链 vocabulary 已冻结，真实 polar 候选按 P attachment 边界生成审核包。
第二层 CORE/EXTENDED 筛选、特殊结构核验和独立 v0.2 组装入口见
[POLAR_PLAUSIBILITY.md](docs/POLAR_PLAUSIBILITY.md)。
正式 baseline 筛选输出在本地 `data/private/pubchem_polar_plausibility_v0.1/`，保持原样；
扩充表在 `data/private/polar_expansion_v0.2/polar_moieties_expanded_v0.2.tsv`，
包含 145,322 个真实 PubChem 来源片段和 49,983 个规则生成 phosphate analog。
新版配置为 `configs/library_build_expanded_v02.yaml`，允许上限 2,000,000；
旧版配置与上限保留。v0.2 默认只检查输入和预算，本轮不运行最终完整库。
后续仍需处理标签对齐、训练/验证/测试划分与检索库使用规则。
当前训练入口仍使用 dummy 数据，接口详见 [DATA_SCHEMA.md](docs/DATA_SCHEMA.md)。
真实训练阶段还需评估库规模并把线性结构检索/内存缓存换成适合该规模的实现。

MS/MS、谱图检索、Spectrum Transformer、微调和完整候选组装全部留待后续讨论，本仓库当前入口不包含这些流程。
