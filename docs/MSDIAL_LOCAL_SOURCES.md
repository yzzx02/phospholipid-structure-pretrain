# 找到的本地 MS-DIAL 原始库

已于 2026-10-13 在本机定位并读取容器头核验以下两个文件。未移动、覆盖或下载数据库。

| 版本线索 | 文件 | 大小 bytes | 容器库记录数 | LZ4 blocks |
|---|---|---:|---:|---:|
| 2025，NCDK conventional converted dev | `D:\MSDIAL\Msp20250409095351_NCDK_conventional_converted_dev.lbm2` | 792,033,100 | 5,263,863 | 4 |
| 2022，MS-DIAL 4.9 下载目录 | `D:\浏览器下载文件\MSDIAL4.9.221218\MSDIAL ver.4.9.221218 Windowsx64\Msp20221205132019_conventional_converted.lbm2` | 372,248,075 | 3,879,270 | 2 |

库记录数来自每个解压块顶层 MessagePack array header。它不是独立分子数，也不是磷脂数；原库包含不同离子形式/重复记录和非磷脂类别，不能把这两个数字直接计入磷脂训练库。

读取方式已对照 MS-DIAL 官方代码：

- [LargeListMessagePack 容器](https://github.com/systemsomicslab/MsdialWorkbench/blob/8556036fb73d77e4d703eb4263defe025d604a5d/src/Common/CommonStandard/MessagePack/LargeListMessagePack.cs)：ext32、type code=99、int32 解压长度和 raw LZ4。
- [v5 MoleculeMsReference 字段定义](https://github.com/systemsomicslab/MsdialWorkbench/blob/8556036fb73d77e4d703eb4263defe025d604a5d/src/Common/CommonStandard/Components/MoleculeMsReference.cs)：仅读取 ScanID、Name、Ontology、SMILES、InChIKey、CompoundClass 的有限样本；Spectrum 字段跳过。

2025 文件每个 block 最多解压前 4 MiB、最多采 32 条元数据，总计 128 条。抽样位置是每块开头，观察到 FAHFA、ASHexCer、Hex3Cer、TG；这不是全面的类别分布或磷脂覆盖率核验。2022 文件仅核验容器和记录计数，没有套用 v5 metadata schema。没有一次性展开每个约 1 GiB 的块，没有导出或分析谱图峰列表。

本地详细核验结果：`data/private/msdial_inventory_2025.json` 和 `data/private/msdial_inventory_2022.json`，不上传原始库或这些源 SMILES 样本。

复跑 2025 核验：

```powershell
python scripts/inspect_msdial_library.py --library 'D:\MSDIAL\Msp20250409095351_NCDK_conventional_converted_dev.lbm2' --v5-metadata --sample-records 32 --output data/private/msdial_inventory_2025.json
```

下一步真实建库要把已有 MS-DIAL 来源整理为 [builder 定义的 chain table](LIBRARY_BUILDER.md)，对 token、唯一代表片段、DB positions 和 P 注释口径做一致性核验。本轮没有从库名称自行猜测这些规则，也没有把原库的完整脂质结构当作 polar head。
