// Flat audit workbook; the companion HTML contains all molecular drawings.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const root=path.resolve(process.argv[2]||'outputs/headgroup_template_proposal_20261022');
const require=createRequire(path.join(root,'authoring','package.json'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload=JSON.parse(await fs.readFile(path.join(root,'authoring/workbook_payload_private.json'),'utf8'));
const summary=payload.summary, coverage=payload.coverage;
const wb=Workbook.create();
const output=wb.worksheets.add('覆盖比较'), review=wb.worksheets.add('补选审核'), known=wb.worksheets.add('已有保留');
function column(i){let x='';for(i++;i;i=Math.floor((i-1)/26))x=String.fromCharCode(65+(i-1)%26)+x;return x}
function base(sheet,cols,rows,widths){
  const end=column(cols-1);sheet.showGridLines=false;
  sheet.getRange(`A1:${end}${rows}`).format={font:{name:'Arial',size:10,color:'#24384A'},verticalAlignment:'center',rowHeight:25};
  widths.forEach((v,i)=>sheet.getRange(`${column(i)}1:${column(i)}${rows}`).format.columnWidthPx=v);
  sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#20384D'};
}
function table(sheet,address,headers,values,name){
  const row=Number(address.match(/\d+/)[0]),end=column(headers.length-1),last=row+values.length;
  sheet.getRange(`A${row}:${end}${row}`).values=[headers];
  sheet.getRange(`A${row}:${end}${row}`).format={fill:'#243C52',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:32};
  if(values.length)sheet.getRange(`A${row+1}:${end}${last}`).values=values;
  const t=sheet.tables.add(`A${row}:${end}${last}`,true,name);t.showFilterButton=true;
  return last;
}
base(output,8,28,[210,160,160,150,150,160,160,180]);
output.tabColor='#243C52';output.getRange('A2').values=[['基础模板数量与核心覆盖']];
output.getRange('A4:D8').values=[['原人工保留记录',summary.approved_catalog_graphs,'质子化归并后',summary.conceptual_human_approved_seed_graphs],['原人工排除',summary.human_decision_counts.exclude,'仍待判断',summary.human_decision_counts.review],['模板特征入口检查通过',summary.current_model_eligible_approved_graphs,'冻结核心总数',195306],['250 个方案新增候选',summary.new_proposals_for_250,'350 个方案新增候选',summary.new_proposals_for_350],['当前 Decoder 标签检查通过',summary.current_decoder_target_eligible_approved_graphs,'去重后模板编码需处理',134-summary.current_model_eligible_approved_graphs]];
output.getRange('A9').values=[['全部核心参与覆盖统计，排除自身及磷酸质子化重复。旧数据划分不限制选模板。']];
const comparison=summary.proposed_sizes.map(size=>{const c=coverage[size].all_frozen;return [size,c.mean_nearest_tanimoto,c['fraction_at_least_0.4'],c['fraction_at_least_0.5'],c['fraction_at_least_0.7'],c.very_far_count,c.formula_distance_quantiles.minimum_heavy_element_L1['0.5'],coverage[size].fixed_unseen_targets['fraction_at_least_0.5']]});
table(output,'A11',['模板数','平均最高相似度','相似度≥0.4','相似度≥0.5','相似度≥0.7','相似度<0.3 个数','重元素计数差中位数','固定非模板目标≥0.5'],comparison,'CoverageComparison');
output.getRange('B12:B16').setNumberFormat('0.000');output.getRange('C12:E16').setNumberFormat('0.0%');output.getRange('H12:H16').setNumberFormat('0.0%');output.getRange('F12:G16').setNumberFormat('#,##0');
output.getRange('A19').values=[['新增候选审核进度']];output.getRange('A20:F20').values=[['保留',null,'排除',null,'待判断',null]];
const additions=payload.rows.filter(r=>r.kind==='new'),approved=payload.rows.filter(r=>r.kind==='approved');
const end=additions.length+6;
output.getRange('B20').formulas=[[`=COUNTIF('补选审核'!D7:D${end},"保留")`]];
output.getRange('D20').formulas=[[`=COUNTIF('补选审核'!D7:D${end},"排除")`]];
output.getRange('F20').formulas=[[`=COUNTIF('补选审核'!D7:D${end},"待判断")`]];
output.getRange('A22').values=[['相似度阈值只用于方案比较，不代表模型预测准确率。']];
output.getRange('A23').values=[['二维结构请打开同目录 index.html。新增候选的 HTML 与 Excel 审核分别保存。']];
base(review,17,end,[80,135,120,100,230,170,110,280,100,100,270,270,520,270,120,360,480]);
review.getRange('A2').values=[['新增模板候选审核']];review.getRange('A3').values=[['黄色列填写决定和备注。全部新增项尚未人工确认；按“从多少个模板起加入”筛选 200/250/300/350 方案。']];
review.freezePanes.freezeRows(6);review.freezePanes.freezeColumns(1);
const vals=additions.map(r=>[r.review_number,r.first_bank_size,r.display_name,'未判断','',r.open_polar_formula,r.average_mass_Da,(r.functional_categories||[]).join('; '),r.nearest_prior_template_tanimoto,r.builder_compatible,r.structure_origin,r.headgroup_id,r.polar_smiles,r.chemistry_family,r.P,r.evidence_source,r.source_attachment_provenance]);
table(review,'A6',['编号','从多少个模板起加入','显示名称','你的决定','你的备注','开放 Polar 分子式','质量 (Da)','官能团','与先前模板相似度','Builder 通过','来源类型','冻结 Polar ID','Polar SMILES','化学家族','P 数','证据来源','连接来源记录'],vals,'NewTemplateProposals');
review.getRange(`D7:E${end}`).format.fill='#FFF2C6';review.getRange(`D7:D${end}`).dataValidation={rule:{type:'list',values:['未判断','保留','排除','待判断']}};
review.getRange(`G7:G${end}`).setNumberFormat('0.0');review.getRange(`I7:I${end}`).setNumberFormat('0.000');review.getRange(`H7:H${end}`).format.wrapText=true;review.getRange(`H7:H${end}`).format.rowHeight=45;
additions.forEach((r,i)=>review.getRange(`A${i+7}:Q${i+7}`).format.rowHeight=Math.max(38,18*Math.ceil((r.functional_categories||[]).join('; ').length/38)));
for(const [value,fill] of [['保留','#E6F2ED'],['排除','#F9E8E5'],['待判断','#FFF1D8']])review.getRange(`D7:D${end}`).conditionalFormats.add('containsText',{text:value,format:{fill}});
const knownEnd=approved.length+6;
base(known,12,knownEnd,[90,210,300,250,340,180,110,115,120,520,450,550]);
known.getRange('A2').values=[['人工保留的原始记录']];known.getRange('A3').values=[['模板编码与目标标签分别检查。84 token 上限只约束 Decoder 输出标签，不限制指纹模板输入。']];known.freezePanes.freezeRows(6);known.freezePanes.freezeColumns(2);
table(known,'A6',['编号','简写或名称','英文名称','原审核备注','模板编码 / 目标标签问题','开放 Polar 分子式','质量 (Da)','Builder 通过','词数含起止符','Polar SMILES','来源','来源链接'],approved.map(r=>[r.review_number,r.abbreviation||r.display_name,r.full_name,r.human_note,[r.training_reference_blockers?'模板: '+r.training_reference_blockers:'',r.target_label_blockers?'标签: '+r.target_label_blockers:''].filter(Boolean).join('; '),r.open_polar_formula,r.average_mass_Da,r.builder_compatible,r.token_length_including_BOS_EOS,r.polar_smiles,r.source,r.source_url]),'ApprovedReference');
known.getRange(`G7:G${knownEnd}`).setNumberFormat('0.0');
known.getRange(`D7:E${knownEnd}`).format.wrapText=true;known.getRange(`D7:E${knownEnd}`).format.rowHeight=45;
approved.forEach((r,i)=>known.getRange(`A${i+7}:L${i+7}`).format.rowHeight=Math.max(38,18*Math.ceil((r.human_note||'').length/17),18*Math.ceil((r.training_reference_blockers||'').length/45)));
// Exercise the editable decision and its dependent summary, then restore it.
review.getRange('D7').values=[['保留']];
if(output.getRange('B20').values[0][0]!==1)throw Error('Decision summary did not recalculate');
review.getRange('D7').values=[['未判断']];
wb.recalculate();
if(output.getRange('B20').values[0][0]!==0)throw Error('Temporary decision was not restored');
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1500});
console.log((await wb.inspect({kind:'table',range:'覆盖比较!A11:H16',include:'values,formulas',tableMaxRows:6,tableMaxCols:8,maxChars:2300})).ndjson);console.log(errors.ndjson);
for(const [sheet,range] of [[output,'A1:H23'],[review,'A1:I12'],[known,'A1:G12']]){
 const blob=await wb.render({sheetName:sheet.name,range,scale:1,format:'png'});
 await fs.writeFile(path.join(root,'authoring',sheet.name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(root,'headgroup_template_proposal.xlsx'));
await fs.writeFile(path.join(root,'authoring/workbook_qa.json'),JSON.stringify({sheets:3,known:approved.length,new:additions.length,decision_change_recalculated:true,temporary_decision_restored:true,error_scan:errors.ndjson},null,2));
console.log(JSON.stringify({exported:'headgroup_template_proposal.xlsx',known:approved.length,new:additions.length}));
