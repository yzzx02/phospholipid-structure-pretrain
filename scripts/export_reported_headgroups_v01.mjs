// Author the review workbook from private CSV-derived JSON with the bundled Artifact Tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const root = path.resolve(process.argv[2] || 'outputs/reported_headgroups_v01_20261020');
const require = createRequire(path.join(root, 'authoring', 'package.json'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload = JSON.parse(await fs.readFile(path.join(root, 'excel_payload_private.json'), 'utf8'));
const wb = Workbook.create();
const query = wb.worksheets.add('名称查询');
const tables = {};
const columns = {
  '头基总表': ['headgroup_id','abbreviation','full_name','polar_smiles','builder_compatible','literature_verified','source_kind','source','source_id_or_doi','source_url','aliases','original_smiles','incompatibility_reason','in_frozen_195306','source_count','name_alias_count','roundtrip_graph_source_count','roundtrip_formula_source_count','review_reason'],
  '名称映射': ['query_lookup_key','query_name','normalized_name','headgroup_id','polar_smiles','name_kind','query_match_index'],
  '来源记录': ['headgroup_id','source_record_id','source','source_id_or_doi','source_url','full_name','abbreviation','original_smiles','polar_smiles','source_kind','msms_status','msms_reference','evidence_locator','boundary_method','roundtrip_graph','roundtrip_formula','roundtrip_reason'],
  '待审核': ['review_id','headgroup_id','abbreviation','full_name','source','source_id_or_doi','review_reason','original_smiles','polar_smiles','source_url'],
};
function column(i) { let out=''; for(i++;i;i=Math.floor((i-1)/26)) out=String.fromCharCode(65+(i-1)%26)+out; return out; }
function literal(v) { return typeof v==='string' && v.startsWith('=') ? "'"+v : v; }
let number = 0;
for (const [name, fields] of Object.entries(columns)) {
  const sheet = wb.worksheets.add(name);
  const rows = payload[name];
  const matrix = [fields, ...rows.map(r => fields.map(k => literal(r[k] ?? '')))];
  const range = sheet.getRange(`A1:${column(fields.length-1)}${matrix.length}`);
  range.values = matrix;
  range.format.font = { name:'Arial', size:10, color:'#20252A' };
  range.format.verticalAlignment = 'center';
  range.format.rowHeight = 20;
  range.format.columnWidth = 24;
  const header = sheet.getRange(`A1:${column(fields.length-1)}1`);
  header.format = { fill:'#243C52', font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'}, horizontalAlignment:'center', verticalAlignment:'center', rowHeight:30 };
  sheet.tables.add(`A1:${column(fields.length-1)}${matrix.length}`, true, `ReportedTable${++number}`);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(name==='头基总表' ? 2 : 1);
  for(let c=0;c<fields.length;c++) {
    const f=fields[c], col=column(c);
    if(['polar_smiles','original_smiles'].includes(f)) sheet.getRange(`${col}1:${col}${matrix.length}`).format.columnWidth = 72;
    if(['full_name','query_name','evidence_locator','review_reason'].includes(f)) sheet.getRange(`${col}1:${col}${matrix.length}`).format.columnWidth = 48;
    if(/count|index/.test(f)) sheet.getRange(`${col}2:${col}${matrix.length}`).setNumberFormat('#,##0');
    if(['headgroup_id','source_record_id','review_id'].includes(f)) sheet.getRange(`${col}1:${col}${matrix.length}`).format.columnWidth = 32;
    if(f==='source_url') sheet.getRange(`${col}1:${col}${matrix.length}`).format.columnWidth = 72;
  }
  if(name==='头基总表') {
    sheet.getRange(`C2:D${matrix.length}`).format.wrapText = true;
    for(let r=0;r<rows.length;r++) {
      const lines=Math.max(Math.ceil((rows[r].polar_smiles || '').length/90),Math.ceil((rows[r].full_name || '').length/45),1);
      sheet.getRange(`A${r+2}:S${r+2}`).format.rowHeight=Math.max(32,lines*14+6);
    }
    sheet.getRange('B1:B'+matrix.length).format.columnWidth = 17;
  } else {
    for(let c=0;c<fields.length;c++) {
      if(['query_lookup_key','query_name','normalized_name','full_name','review_reason','source_url','source_id_or_doi','original_smiles','polar_smiles'].includes(fields[c])) {
        sheet.getRange(`${column(c)}2:${column(c)}${matrix.length}`).format.wrapText = true;
      }
    }
    range.format.autofitRows();
  }
  tables[name]={sheet, rows:rows.length, fields};
}
const m=tables['名称映射'].rows+1, h=tables['头基总表'].rows+1;
query.getRange('A1:H20').format.font={name:'Arial',size:10,color:'#20252A'};
query.getRange('A1:H20').format.verticalAlignment='center';
query.getRange('A1:H20').format.rowHeight=24;
query.getRange('A1:A20').format.columnWidth=24;
query.getRange('B1:B20').format.columnWidth=85;
query.getRange('C1:H20').format.columnWidth=12;
query.showGridLines=false;
query.getRange('A2:B2').values=[['Polar 头基名称查询','v0.1 来源快照']];
query.getRange('A2:B2').format.font={name:'Arial',size:14,bold:true};
query.getRange('A3:A11').values=[['名称 / 简写'],['标准化查询'],['匹配序号（可修改）'],['匹配结构总数'],['状态'],['Polar ID'],['Polar SMILES'],['英文名称'],['Builder 兼容']];
query.getRange('B3').values=[['PC']];
query.getRange('B5').values=[[1]];
query.getRange('B3').format.fill='#FFF1BB';
query.getRange('B5').format.fill='#FFF1BB';
query.getRange('B4').formulas=[['=LOWER(TRIM(B3))']];
query.getRange('B6').formulas=[[`=COUNTIFS('名称映射'!$C$2:$C$${m},B4)`]];
query.getRange('B7').formulas=[['=IF(B6=0,"未找到",IF(OR(B5<1,B5>B6),"匹配序号超出范围",IF(B6>1,"同名多结构：修改匹配序号逐一查看","唯一结构")))']];
query.getRange('B8').formulas=[[`=IF(OR(B6=0,B5<1,B5>B6),"",VLOOKUP(B4&"|"&B5,'名称映射'!$A$2:$G$${m},4,FALSE))`]];
query.getRange('B9').formulas=[[`=IF(B8="","",VLOOKUP(B4&"|"&B5,'名称映射'!$A$2:$G$${m},5,FALSE))`]];
query.getRange('B10').formulas=[[`=IF(B8="","",VLOOKUP(B8,'头基总表'!$A$2:$S$${h},3,FALSE))`]];
query.getRange('B11').formulas=[[`=IF(B8="","",IF(VLOOKUP(B8,'头基总表'!$A$2:$S$${h},5,FALSE),"是","否"))`]];
query.getRange('B9').format.wrapText=true;
query.getRange('A9:B9').format.rowHeight=110;
query.getRange('A13:B16').values=[['查询方法','修改黄色单元格。PPA、LPG 等同名多结构时，修改匹配序号查看全部结果。'],['反查来源','在名称映射或来源记录工作表筛选 Polar ID，可反查全部名称和来源。'],['来源说明','literature_verified 仅表示已核实头基文献；其余数据库或规则结构明确标为未核实。'],['待审核记录','特殊骨架、名称冲突和无法可靠确定连接位点的记录均保留；未为了兼容而修改重原子图。']];
query.getRange('B13:B16').format.wrapText=true;
query.getRange('A13:B16').format.rowHeight=38;
query.dataValidations.add({range:'B5',rule:{type:'whole',operator:'between',formula1:1,formula2:1000}});
const notes=wb.worksheets.add('说明');
notes.showGridLines=false;
const explanation=[['项目','已报道及数据库来源含 P 头基参考库 v0.1'],['结构记录',payload.summary.distinct_resolved_polar_graphs],['文献已核实',payload.summary.literature_verified_graphs],['文献尚未核实',payload.summary.source_database_only_or_literature_unverified_graphs],['Builder 通过',payload.summary.builder_compatible],['冻结库中已存在',payload.summary.in_frozen_195306],['待审核行',payload.summary.pending_review_rows],['身份定义','以连接 P 为根、带 [*:1] 的非手性 canonical graph；原始立体结构保留在来源记录。'],['计数边界','CL 等特殊骨架切下的残余图仍可能包含其他磷脂酰臂，作为独立来源图保留，不代表同样数量的基础核心。'],['文献边界','头基文献链接不代表每种数据库完整脂质链组合都已获实验确认。'],['谱图','experimental/theoretical/unknown 表示来源信息；本轮没有生成或搜索谱图。'],['完整性','CSV 是完整机器数据；本工作簿保留全部来源行，并仅展示人工检查需要的列。'],['后续','不改冻结划分、模型、Tokenizer 或 Builder；本轮未训练。']];
notes.getRange(`A1:B${explanation.length}`).values=explanation;
notes.getRange(`A1:B${explanation.length}`).format.font={name:'Arial',size:10};
notes.getRange(`A1:A${explanation.length}`).format.columnWidth=24;
notes.getRange(`B1:B${explanation.length}`).format.columnWidth=100;
notes.getRange(`B1:B${explanation.length}`).format.wrapText=true;
notes.getRange(`A1:B${explanation.length}`).format.rowHeight=32;
notes.getRange('A1:B1').format.font={name:'Arial',size:14,bold:true};

// Check real name ambiguity and an out-of-range selection before restoring the default.
query.getRange('B3').values=[['PPA']];
const collision=query.getRange('B6').values[0][0];
if(collision!==2) throw new Error(`Expected two PPA graphs, got ${collision}`);
query.getRange('B5').values=[[1]];
const first=query.getRange('B8').values[0][0];
query.getRange('B5').values=[[2]];
const second=query.getRange('B8').values[0][0];
const expectedPPA=payload['名称映射'].filter(r=>r.normalized_name==='ppa').sort((a,b)=>a.query_match_index-b.query_match_index);
if(first!==expectedPPA[0].headgroup_id || second!==expectedPPA[1].headgroup_id) throw new Error('Ambiguous-name selector returned incorrect graphs');
query.getRange('B3').values=[['not_a_lipid_headgroup']];
if(query.getRange('B6').values[0][0]!==0 || query.getRange('B9').values[0][0]!=='') throw new Error('No-match lookup failed');
query.getRange('B3').values=[['PC']];
query.getRange('B5').values=[[1]];
wb.recalculate();
const expectedPC=payload['名称映射'].find(r=>r.normalized_name==='pc' && r.query_match_index===1);
if(query.getRange('B8').values[0][0]!==expectedPC.headgroup_id || query.getRange('B9').values[0][0]!==expectedPC.polar_smiles) throw new Error('PC lookup returned an incorrect graph');
console.log((await wb.inspect({kind:'table',range:'名称查询!A3:B11',include:'values,formulas',tableMaxRows:9,tableMaxCols:2,maxChars:2200})).ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'Formula error check',maxChars:2000});
console.log(errors.ndjson);
const preview=path.join(root,'authoring','previews');
await fs.mkdir(preview,{recursive:true});
for(const [sheetName,range,label] of [['名称查询','A1:B16','Query'],['头基总表','A1:F7','Headgroups'],['名称映射','A1:D7','Names'],['来源记录','A1:F6','Sources'],['待审核','A1:G6','Pending'],['说明','A1:B13','Notes']]) {
  const blob=await wb.render({sheetName,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(preview,label+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(root,'reported_headgroups_v01.xlsx'));
await fs.writeFile(path.join(root,'authoring','workbook_qa.json'),JSON.stringify({PPA_matches:collision,distinct_selector_ids:true,no_match_returns_blank:true,sheets:6,rows:Object.fromEntries(Object.entries(tables).map(([k,v])=>[k,v.rows])),default_name:'PC',formula_errors_inspection:errors.ndjson},null,2));
console.log(JSON.stringify({saved:path.join(root,'reported_headgroups_v01.xlsx'),sheets:6}));
