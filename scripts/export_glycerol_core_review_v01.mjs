// One row per connectivity core. Complete structures and provenance remain on G.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const root=path.resolve(process.argv[2]||'outputs/glycerol_core_collection_v01_20261023');
const require=createRequire(path.join(root,'authoring','package.json'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload=JSON.parse(await fs.readFile(path.join(root,'authoring/workbook_payload_private.json'),'utf8'));
const rows=payload.cores;
const wb=Workbook.create();
const review=wb.worksheets.add('核心审核'),sources=wb.worksheets.add('来源索引');
const category={natural_reported_core:'天然核心文献依据',database_whole_lipid_natural_unverified:'完整脂质库记录；天然待核实',reported_synthetic_core:'有合成记录；天然待核实',natural_linkage_rule_candidate:'天然连接规则扩展候选',source_definition_or_reference_natural_unverified:'规则或旧参考；天然待核实'};
const column=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s};
function base(sheet,widths,last){sheet.showGridLines=false;sheet.getRange(`A1:${column(widths.length-1)}${last}`).format={font:{name:'Arial',size:10,color:'#24384A'},rowHeight:27,verticalAlignment:'center'};widths.forEach((w,i)=>sheet.getRange(`${column(i)}1:${column(i)}${last}`).format.columnWidthPx=w)}
function table(sheet,start,headers,values,name){const last=start+values.length,end=column(headers.length-1);sheet.getRange(`A${start}:${end}${start}`).values=[headers];if(values.length)sheet.getRange(`A${start+1}:${end}${last}`).values=values;sheet.tables.add(`A${start}:${end}${last}`,true,name);sheet.getRange(`A${start}:${end}${start}`).format={fill:'#263F52',font:{name:'Arial',size:10,color:'#FFFFFF',bold:true},rowHeight:34,horizontalAlignment:'center',verticalAlignment:'center'};sheet.freezePanes.freezeRows(start);sheet.freezePanes.freezeColumns(2);return last}
const last=rows.length+8;
base(review,[255,255,230,105,260,125,145,110,110,110,140,340,570,570,300,480,480],last);
review.tabColor='#263F52';review.getRange('A2').values=[['甘油＋头基结构审核']];review.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#263F52'};
review.getRange('A3').values=[['每个连接图一个 ID；立体异构合并。黄色列填写本轮决定；旧人工决定单独保留。二维结构见同目录 index.html。']];
review.getRange('A4:L5').values=[['结构通过',null,'','天然文献核心',null,'','待判断',null,'','初筛排除',null,''],['本轮保留',null,'','本轮排除',null,'','本轮待判断',null,'','','','']];
for(const [cell,col,value] of [['B4','F','结构检查通过'],['E4','C','天然核心文献依据'],['H4','F','待判断'],['K4','F','排除候选'],['B5','D','保留'],['E5','D','排除'],['H5','D','待判断']])review.getRange(cell).formulas=[[`=COUNTIFS(${col}9:${col}${last},"${value}")`]];
review.getRange('A6').values=[['质量按链位点 H 封端核心：糖基或肌醇 600 Da，其余 400 Da。天然证据不证明来源库的每一种酰基链组合。']];
const headers=['核心 ID','名称／简写','来源性质','本轮决定','本轮备注','结构筛选','核心分子式','核心质量 (Da)','质量上限 (Da)','实际酰基来源数','原人工决定','筛选原因','甘油＋头基 SMILES','带链位点显示 SMILES','旧头基编号','天然文献链接','代表结构来源'];
const values=rows.map(r=>[r.core_id,r.display_name.slice(0,180),category[r.evidence_category],r.human_decision,r.human_note,r.selection_status,r.core_formula,r.core_average_mass_Da,r.mass_limit_Da,r.actual_acyl_lipid_source_count,r.manual_decisions.join('; '),r.proposed_filter_reasons.join('; '),r.core_smiles,r.display_smiles,r.legacy_headgroup_ids.join('; '),r.natural_core_evidence.map(x=>x.source_url).join('; '),r.source_urls[0]||'']);
table(review,8,headers,values,'GlycerolCoreReview');
review.getRange(`D9:E${last}`).format.fill='#FFF1C5';review.getRange(`D9:D${last}`).dataValidation={rule:{type:'list',values:['未判断','保留','排除','待判断']}};
review.getRange(`H9:H${last}`).setNumberFormat('0.0');review.getRange(`I9:J${last}`).setNumberFormat('0');
review.getRange(`B9:C${last}`).format.wrapText=true;review.getRange(`K9:L${last}`).format.wrapText=true;
for(let i=0;i<rows.length;i++)review.getRange(`A${i+9}:Q${i+9}`).format.rowHeight=Math.max(42,Math.min(96,18*Math.ceil(Math.max(rows[i].display_name.slice(0,180).length/28,rows[i].proposed_filter_reasons.join('; ').length/48))));
for(const [text,fill] of [['保留','#E5F2E8'],['排除','#FAE4DF'],['待判断','#FFF1CC']])review.getRange(`D9:D${last}`).conditionalFormats.add('containsText',{text,format:{fill}});
const provenance=payload.representative_sources.map(r=>[r.core_id,r.source,r.source_id,r.source_kind,r.full_lipid_with_chains,r.actual_acyl_chains_present,r.extraction.parent_roundtrip_connectivity===true?'通过':r.full_lipid_with_chains?'通过':'不适用：来源为规则／参考',r.extraction.parent_roundtrip_formula_including_H===true?'通过':'不适用：来源为规则／参考',r.name,r.original_smiles,r.source_url,'完整来源链组合的天然存在尚未逐条核实']);
for(const e of payload.literature)provenance.push([e.core_id,'原始文献',e.pmid?'PMID:'+e.pmid:e.doi,e.source_kind,false,false,'核心证据；非逐条链组合','不适用',e.name,'',e.source_url,e.scope]);
base(sources,[255,140,220,330,115,115,220,240,400,600,540,720],provenance.length+1);
table(sources,1,['核心 ID','来源','来源编号','来源类型','完整脂质图','含实际酰基链','源图连接往返','源图分子式含 H','源名称','原始来源 SMILES／表达式','来源链接','证据范围'],provenance,'CoreProvenance');
sources.getRange(`A2:L${provenance.length+1}`).format.rowHeight=32;
review.getRange('D9').values=[['保留']];if(review.getRange('B5').values[0][0]!==1)throw Error('Editable decision did not recalculate');review.getRange('D9').values=[['未判断']];
wb.recalculate();if(review.getRange('B5').values[0][0]!==0)throw Error('Decision test was not restored');
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:8},maxChars:1500});
console.log((await wb.inspect({kind:'table',range:'核心审核!A4:K5',include:'values,formulas',tableMaxRows:2,tableMaxCols:11,maxChars:2200})).ndjson);console.log(errors.ndjson);
for(const [sheet,range,name] of [[review,'A1:I13','review_preview.png'],[sources,'A1:F9','sources_preview.png']]){const blob=await wb.render({sheetName:sheet.name,range,scale:1,format:'png'});await fs.writeFile(path.join(root,'authoring',name),new Uint8Array(await blob.arrayBuffer()))}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(root,'glycerol_core_review_v01.xlsx'));
await fs.writeFile(path.join(root,'authoring/workbook_qa.json'),JSON.stringify({core_rows:rows.length,source_audit_rows:provenance.length,editable_decision_recalculates:true,temporary_decision_restored:true,error_scan:errors.ndjson},null,2));
// The compact QA JSON is retained; the exporter's duplicate cell dump is rebuildable.
await fs.unlink(path.join(root,'glycerol_core_review_v01.xlsx.inspect.ndjson')).catch(error=>{if(error.code!=='ENOENT')throw error});
console.log(JSON.stringify({workbook:'glycerol_core_review_v01.xlsx',core_rows:rows.length,source_audit_rows:provenance.length}));
