"""Independent, private audit and aggregate-only publication of template proposals."""

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs

from structure_pretrain.reported_headgroups.template_selection import (
    digest, fp, load_review, phosphate_comparison_key, read_csv, write_csv, methylene_contractions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/headgroup_template_proposal_20261022'))
    args = parser.parse_args()
    out = args.output
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    coverage = json.loads((out/'coverage_summary.json').read_text(encoding='utf-8'))
    baseline = json.loads((out/'selection_baselines_250.json').read_text(encoding='utf-8'))
    banks = json.loads((out/'proposed_banks_private.json').read_text(encoding='utf-8'))
    raw, review = load_review(Path('outputs/reported_headgroups_v02_20261021/reported_headgroups_v02.csv'),
                             out/'human_review_snapshot/headgroup_manual_review.json',
                             out/'human_review_snapshot/headgroup_manual_review.csv')
    approved = read_csv(out/'approved_reference_catalog.csv')
    assert {r['headgroup_id'] for r in approved} == {k for k,d in review['decisions'].items() if d['decision']=='keep'}
    assert len(approved) == 135
    for row in approved:
        assert row['human_note'] == review['decisions'][row['headgroup_id']].get('note','')
    previous = set()
    for size, rows in banks.items():
        assert len(rows) == int(size)
        keys = {phosphate_comparison_key(r['polar_smiles']) for r in rows}
        assert len(keys) == len(rows)
        assert previous <= keys
        previous = keys
    rows = banks['350']
    bank_keys = {phosphate_comparison_key(r['polar_smiles']):r for r in rows}
    homologs = []
    for row in rows:
        for shorter in methylene_contractions(row['polar_smiles']):
            other = bank_keys.get(shorter)
            if other and (row['selection_status'] != 'human_approved' or other['selection_status'] != 'human_approved'):
                homologs.append({'headgroup_id_1': row['headgroup_id'], 'headgroup_id_2':other['headgroup_id'],
                                 'larger_bank_size_required': max(row.get('first_bank_size',0),other.get('first_bank_size',0)),
                                 'relationship':'one_nonring_CH2_insertion_exact_graph', 'automatic_deletion':False})
    write_csv(out/'methylene_homolog_pairs_for_review.csv', homologs,
              ['headgroup_id_1','headgroup_id_2','larger_bank_size_required','relationship','automatic_deletion'])
    original_keys = {phosphate_comparison_key(r['polar_smiles']) for r in raw}
    fingerprints = []
    for row in rows:
        fingerprint = fp(row['polar_smiles'])
        if row['selection_status'] != 'human_approved':
            assert phosphate_comparison_key(row['polar_smiles']) not in original_keys
            assert row['human_decision'] == ''
            assert row['average_mass_Da'] <= 600
            assert row['builder_compatible']
            maximum = max(DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints))
            assert maximum <= .85
            assert abs(maximum-row['nearest_prior_template_tanimoto']) < 1e-12
        fingerprints.append(fingerprint)
    for filename, expected in summary['source_hashes'].items():
        assert digest(filename) == expected
    visual = json.loads((out/'visual_validation.json').read_text())
    assert visual['graphs'] == visual['svg'] == visual['png'] == 351
    for path in (out/'svg').glob('*.svg'):
        ET.parse(path)
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(out/'headgroup_template_proposal.xlsx') as archive:
        book = ET.fromstring(archive.read('xl/workbook.xml'))
        assert [s.attrib['name'] for s in book.find('s:sheets',ns)] == ['覆盖比较','补选审核','已有保留']
        review_sheet = ET.fromstring(archive.read('xl/worksheets/sheet2.xml'))
        assert len(review_sheet.findall('s:sheetData/s:row',ns)) >= 216
        assert review_sheet.find('s:dataValidations',ns) is not None
    rows = banks['250']
    element_sets = coverage['250']['all_frozen']['element_sets']
    missing = [{'element_set': k, **v} for k,v in element_sets.items() if v['template_count']==0]
    write_csv(out/'uncovered_element_combinations_250.csv', missing)
    descriptors = pq.read_table('outputs/polar_size_scope_audit_20261022/per_graph_descriptors_private.parquet').to_pylist()
    by_id = {r['polar_id']:r for r in descriptors}
    strata = []
    for size in summary['proposed_sizes']:
        metric = pq.read_table(out/f'target_coverage_bank_{size}_private.parquet',columns=['polar_id','nearest_tanimoto']).to_pydict()
        grouped = {}
        for identifier, sim in zip(metric['polar_id'],metric['nearest_tanimoto']):
            d = by_id[identifier]
            label = f"P{d['P']}/mass_{int(d['average_mass_Da']//150)*150}_{(int(d['average_mass_Da']//150)+1)*150}"
            grouped.setdefault(label,[]).append(sim)
        for label, values in sorted(grouped.items()):
            templates = [r for r in banks[str(size)] if (f"P{r['P']}/mass_{int(r['average_mass_Da']//150)*150}_{(int(r['average_mass_Da']//150)+1)*150}") == label]
            strata.append({'bank_size':size,'formula_mass_P_stratum':label,'target_count':len(values),
                           'template_count':len(templates),'fraction_tanimoto_at_least_0.5':float(np.mean(np.asarray(values)>=.5))})
    write_csv(out/'formula_mass_P_stratum_coverage.csv',strata)
    audit = {'status':'passed','human_review_JSON_CSV_source_agree':True,'all_135_keep_graphs_and_notes_preserved':True,
             'nested_banks_and_unique_protonation_groups':True,'all_new_exact_pairwise_similarity_at_most_0.85':True,
             'all_new_mass_at_most_600':True,'new_candidates_not_in_original_324':True,
             'all_original_input_hashes_unchanged':True,'structure_drawings':351,'xlsx_sheets':3,
             'native_XLSX_validation_present':True,'training_started':False}
    audit['exact_single_CH2_homolog_pairs_involving_additions_250'] = sum(r['larger_bank_size_required']<=250 for r in homologs)
    audit['exact_single_CH2_homolog_pairs_involving_additions_350'] = len(homologs)
    (out/'validation_private.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    public_summary = {k:v for k,v in summary.items() if k!='source_hashes'}
    public_summary['source_sha256'] = {Path(k).name:v for k,v in summary['source_hashes'].items()}
    public = {'summary':public_summary,
              'coverage':{n:{k:v for k,v in values.items() if k in ('all_frozen','fixed_unseen_targets')}
                          for n,values in coverage.items()},
              'baselines_250':baseline,'audit':audit}
    results = Path('docs/results')
    (results/'HEADGROUP_TEMPLATE_PROPOSAL_20261022.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = [
        '# 已审核头基整合与固定模板库补选（2026-10-22）', '',
        '已完成 CPU 有界选样与全库覆盖评价。没有训练，也没有自动将新增候选写入正式模板库。', '',
        '用户随后明确：本轮从完整冻结库选基础模板，旧训练/验证/测试划分不限制选样；旧文件、权重和报告原样保留。旧划分结果不能作为这个新模板库的独立泛化评价。', '',
        '## 人工审核整合', '',
        '- 输入 324 个源图，人工保留 135、排除 3、待判断 2；其余 184 没有新增人工决定。JSON 与 CSV 的源哈希、ID、图、决定和备注逐项一致。',
        '- 135 个保留记录中，一组仅终端磷酸氧质子化不同，按比较键归并为 134 个基础模板。另一个待判断记录与既有 PC 等价，追加名称/来源映射，不自动变成人工保留。原图和决定均保存。',
        '- 26 对已保留图的 Morgan 相似度 ≥0.85，列为可选人工精简项，不自动删除连接异构体或同系物。',
        '- 127/135 通过现有 Builder，并用现有 v0.4 代码完成单链/双链无链甘油–Polar 特征入口检查；这 127 个可构造模板特征。134 个概念模板中有 7 个仍受 Builder/电荷/自由基入口限制；250 个概念方案中 243 个通过前置模板特征检查，尚未接入新训练缓存。',
        '- 生成目标另作检查：按原词表及含 BOS/EOS 的长度 84，120 个已确认记录可作当前 Decoder 的输出标签。7 个较长结构超长；长度限制只约束目标标签，不限制作为指纹模板输入。模板输入兼容和输出标签兼容不能混为一谈。', '',
        '## 选择方法', '',
        '从冻结 195,306 图开始，排除已在本次 324 图来源库中的 74 个图，以及其余明显额外脂质锚点 4,752 个，候选池 190,480。新增保持原 600 Da 上限，不另加 C/N/S 数量或卤素禁令。已知参考库的较大真实头基不因这条新增候选上限被删除。', '',
        '1. 保留正确的 P–[*:1] 边界，现有 RDKit/Builder 检查选中候选，并独立核对组装后的全部元素（包括 H）。不从任意首个 P 重新选择连接点。原始出处的连接证据另存；计算有效性不等于已证实生物合成或存在报道。',
        '2. Morgan radius2/2048 位、无手性、保留连接标记。多样性提案加固定随机种子的均匀候选构成约 4,200 个代表，使用 4,096 个全池均匀探针优化设施选址式覆盖收益。',
        '3. 评分为探针平均最高相似度收益 60%、Morgan 差异 25%、缩放元素计数差异 10%、尚未覆盖官能团 5%；按 150 Da 质量区间×P 数给予最高 1.25 倍欠覆盖奖励。权重是首轮可复现的工程选择，没有作为最优超参数宣称。',
        '4. 每个新增模板与此前全部模板的最高 Tanimoto ≤0.85，减少大量相近同系物重复；此阈值不能保证绝无 CH₂ 同系物。16 个公开 SMARTS 类别涵盖胺、羟基、羧基、腈基、酰胺、酯、醚、杂环及特殊 P 键等。',
        '5. 200/250/300/350 为同一排序的嵌套前缀。选样后对全部 195,306 个图精确重算距离；排除目标自身及其磷酸质子化等价组，不按指纹相同合并结构。', '',
        '## 全库结果', '',
        '下表仅为模板覆盖诊断；0.3/0.4/0.5 是操作性阈值，均不是鉴定准确率，也不是普遍化学阈值。', '',
        '|模板数|新增待审|最近相似度 ≥0.4|最近相似度 ≥0.5|最近相似度 <0.3 数量|最近重元素计数差中位数|',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for n in summary['proposed_sizes']:
        c=coverage[str(n)]['all_frozen']
        lines.append(f"|{n}|{max(0,n-134)}|{c['fraction_at_least_0.4']:.2%}|{c['fraction_at_least_0.5']:.2%}|{c['very_far_count']:,}|{c['formula_distance_quantiles']['minimum_heavy_element_L1']['0.5']:.0f}|")
    lines += ['', '250 模板的随机补选对照（3 个固定种子）相似度 ≥0.5 覆盖为 13.05%–13.49%；只追求多样性的对照为 8.49%。最终覆盖感知方案为 22.89%。同一未知目标集合（排除所有 350 模板及已知模板的目标组）对应覆盖为 22.87%，结论不依赖选中模板自匹配。', '',
              f"250 个方案覆盖所有 10 种原库元素和全部 14 种在目标池出现的官能团类别；但仍缺少 {len(missing)} 种精确元素集合组合，涉及 {sum(r['target_count'] for r in missing):,} 个目标。主要涉及少见卤素混合与部分含硫组合；完整组合表留 G 盘。类别里存在模板不表示该类别所有连接方式均已覆盖。", '',
              f"额外进行非环二连接 CH₂ 删除/连边的精确图检查：250 个方案中涉及补选项的单 CH₂ 同系物对为 {audit['exact_single_CH2_homolog_pairs_involving_additions_250']}，350 个方案为 {len(homologs)}。已单独列作审核项，未为凑多样性而自动删除。该检查只覆盖这种明确插入关系，不声称穷尽所有同系关系。", '',
              '开放 Polar 的分子式按相同 [*:1] 边界计数，质量包含图上的显式/隐式 H，不将连接标记视为元素。分别保存最近模板结构相似度、任意模板最小全元素 L1（含 H）、最小重元素 L1、缩放分子式差异及结构最近模板的全元素差异。分子式相近不等于图相近。', '',
              '新增到 250 的 116 图包括 PubChem baseline 51、multi-attachment split 32、程序生成 phosphate analog 33；没有升级为“已报道磷脂”。全套到 350 的 216 个候选均通过现有 Builder 和独立组装元素/H 校验，等待看图审核。', '',
              '## 建议与训练含义', '',
              '先以 250 为审核起点，资源允许时比较 300。250→300 的 ≥0.5 覆盖提高 2.30 个百分点，300→350 再提高 1.82 个百分点，收益递减但尚未消失。350 仍有 24,376 图相似度 <0.3，不宜宣称几百个模板已充分覆盖全部核心。', '',
              '几百个结构作为固定条件模板库，较大核心池作为生成训练目标，方向与结构到结构预训练相容。当前模型每个目标只读取检索出的少量模板（现配置 topK3），不是将整个库全部同时输入。正式启动前需固定人工审核后的模板库、处理 7 个模板入口兼容问题，构建新的独立模板特征/检索缓存，再定义新实验目标集合与独立评价；本轮不启动训练或 MS/MS。', '',
              '参考 [MetGenX 原论文](https://www.nature.com/articles/s41467-026-72149-6)和[作者仓库](https://github.com/ZhuMetLab/MetGenX)：结构到结构预训练后再进行谱图阶段训练。本文的固定 200–350 模板设计是本项目的适配方案，不宣称原文采用相同数量、检索规则或指纹。', '',
              '## 文件与复现', '',
              '完整结构、人工决定、来源别名映射、HTML、SVG、PNG、Excel、逐目标距离及输入快照均保存在 G 盘 `outputs/headgroup_template_proposal_20261022/`。Git 仅提交脚本、测试及本聚合 Markdown/JSON。', '',
              '```powershell', r'. .\scripts\enter_cuda_v04.ps1', r'python scripts\propose_headgroup_template_bank.py', r'python scripts\render_headgroup_template_proposal.py', '# 使用 bundled Artifact Tool 生成 Excel 后', r'python scripts\audit_headgroup_template_proposal.py', r'python -m pytest tests\reported_headgroups\test_template_selection.py -q', '```', '',
              '验证：严格输入哈希与审核表一致性、端磷酸质子化比较、P=S/P=C/多 P 根保留、自匹配排除、嵌套选择、逐新增模板相似度上限、351 张 SVG XML、工作簿原生结构/校验列表、公式重算与三个工作表视觉检查通过。浏览器已检查搜索、统计页和审核标记恢复。旧权重、冻结库、数据划分、Builder 和标签未改动。', '']
    report='\n'.join(lines)
    (results/'HEADGROUP_TEMPLATE_PROPOSAL_20261022.md').write_text(report,encoding='utf-8')
    (out/'headgroup_template_summary.md').write_text(report,encoding='utf-8')
    print(json.dumps({'audit':audit,'missing_element_sets':len(missing),'affected_targets':sum(r['target_count'] for r in missing)},ensure_ascii=False))


if __name__=='__main__':
    main()
