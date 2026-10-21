# 单连接 Polar 结构初筛与人工看图审核

基于 v0.1 的 {input_graphs} 个不同 Polar 图，原始总库保持不变。

- 初筛保留：{retain}
- 明确排除：{exclude}
- 待人工判断：{review}
- 默认浏览：保留与待判断合计 {remaining} 个；排除项另页可查看。

打开 index.html 即可离线看图。点击结构放大，用左右键逐张查看，
标记保留/排除/待判断后导出 CSV 或 JSON。
浏览器本地缓存可能因浏览器或页面路径变化失效，请保存 JSON 备份。
Excel 的人工决定和备注是独立输入，修改 Excel 不会自动改写 HTML。

## 规则

直接检查已定位的 Polar graph，不根据原始完整脂质名称中出现的酰基长链误删小头基。
CL/BMP/LBPA/SLBPA/DEACL/BPG、NAPE/NAPS 类型按本轮范围排除。
图中连续至少 8 个仅有 C/H 邻接的非环脂肪碳视为明显额外长链；
甾醇四稠环也作为额外脂质锚点排除。古菌醚链等额外疏水锚点同样排除，
不能仅依赖 Builder 是否通过。

5–7 碳疏水片段、酯/酰胺取代但无明确长链、来源名称冲突，
以及没有明确长链但 Builder 不兼容的项进入待判断。
羧酸不是酰基长链证据；PIP 多 P、PIM 多糖环本身不触发排除。
短乙酰基和 Ala/Lys-PG 不冒称为额外脂肪酸长链，供你看图决定。

没有为了兼容而切除、修复或改写任何原子和键。
图中 [*:1] 与其相邻的 P 用蓝色标出；红色仅提示算法发现的长链或甾醇原子，
不代表完整的化学官能团边界。
所有图沿用 v0.1 非手性模型表示，原始立体信息保留在源库。

## 文件

- index.html：独立离线浏览页，包含全部内嵌 SVG；默认只展示保留和待判断。
- svg/ 与 png/：按稳定 Polar ID 保存的单结构图（含排除项，便于复核）。
- atlas_svg/ 与 atlas_png/：每页六个结构。
  remaining 为默认图册，excluded 为排除项图册。
- headgroup_structure_review.xlsx：带嵌入结构图的审核表，黄色列填写人工决定与备注。
- headgroup_filter_review.csv：全部 {input_graphs} 个图的逐条决定、原因和来源。
- headgroups_retained.csv、headgroups_excluded.csv、headgroups_manual_review.csv：
  三个互斥初筛集合。

文献已核实/未核实标记沿用 v0.1；结构过滤不会提升来源证据等级。
未加入 v0.1 中尚无可靠 Polar 图的 620 条来源记录。
本轮未训练或自动将人工决定写回正式模板库。
