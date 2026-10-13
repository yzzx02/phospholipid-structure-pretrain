# 示例数据说明

本目录仅包含明确标注的合成 fixture，供验证独立 builder 的化学连接、预算、采样与输出接口。

- 11 个 chain fragments 和它们的双键位置为示例常量，**不是从找到的 MS-DIAL 库导入的真实规则**。
- PC、PE、phosphonate 和 synthetic expanded polar 的 `source` 均为 demo；synthetic expanded 不代表已抓取 PubChem。
- `p_to_o_mapping.tsv` 是显式给出的示例映射，不代表 MS-DIAL 真实定义。
- demo P token 采用外部声明 `p_db_count_includes_vinyl=false`，其真实 vinyl 位置仍保存为 `[1]`。
- `demo_build.yaml`：max_structures=100、max_extra_per_head=8、seed=42。当前实际生成 60 个唯一完整分子图，lyso 6 个。
- `library_report.example.json` 为实际 demo build 的完整报告，用于展示字段。

这些数据和生成结果不能替代真实磷脂训练库。完整流程与接口见 [LIBRARY_BUILDER.md](../../docs/LIBRARY_BUILDER.md)。
