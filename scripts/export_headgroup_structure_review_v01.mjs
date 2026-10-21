// Build an image-first review workbook using the bundled Artifact Tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const root=path.resolve(process.argv[2]||'outputs/reported_headgroups_single_anchor_review_20261021');
const require=createRequire(path.join(root,'authoring','package.json'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload=JSON.parse(await fs.readFile(path.join(root,'authoring/review_payload_private.json'),'utf8'));
const wb=Workbook.create();
const labels={retain:'初筛保留',review:'待人工判断',exclude:'初筛排除'};
const groups=[['看图审核',payload.rows.filter(r=>r.auto_status!=='exclude')],['排除项复核',payload.rows.filter(r=>r.auto_status==='exclude')]];
const fields=['review_number','abbreviation','drawing','human_decision','human_note','auto_status','filter_reason','headgroup_id','full_name','polar_smiles','builder_compatible','literature_verified','source','source_url'];
const headers=['编号','简写','二维结构','你的决定','你的备注','初筛结果','初筛原因','Polar ID','原始英文名称','Polar SMILES','Builder 通过','文献已核实','来源','来源链接'];
const widths=[70,170,470,115,220,120,360,250,420,620,115,115,180,500];
function col(i){let t='';for(i++;i;i=Math.floor((i-1)/26))t=String.fromCharCode(65+(i-1)%26)+t;return t}
const sheets=groups.map(([name,rows])=>({sheet:wb.worksheets.add(name),rows}));
let images=0;
for(const {sheet,rows} of sheets){
  const end=rows.length+6;
  const range=sheet.getRange(`A1:N${end}`);
  range.format.font={name:'Arial',size:10,color:'#24384A'};
  range.format.verticalAlignment='center';
  range.format.rowHeight=24;
  sheet.showGridLines=false;
  sheet.freezePanes.freezeRows(6);
  sheet.freezePanes.freezeColumns(2);
  sheet.tabColor=sheet.name==='看图审核'?'#254C69':'#896A50';
  widths.forEach((v,i)=>sheet.getRange(`${col(i)}1:${col(i)}${end}`).format.columnWidthPx=v);
  sheet.getRange('A2').values=[[sheet.name==='看图审核'?'单连接 Polar 头基结构审核':'额外脂质锚点排除项复核']];
  sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#20384D'};
  sheet.getRange('A3').values=[['黄色列填写你的决定和备注。蓝色为 P-[*:1] 连接位点；红色为额外长链或甾醇锚点。']];
  sheet.getRange('A3').format.font={name:'Arial',size:10,color:'#526273'};
  sheet.getRange('A5').values=[['图按编号固定排列；需要搜索或分类浏览请使用 HTML。Excel 和 HTML 的人工决定分别保存。']];
  sheet.getRange('A5').format.font={name:'Arial',size:10,color:'#526273'};
  sheet.getRange('A6:N6').values=[headers];
  sheet.getRange('A6:N6').format={fill:'#243C52',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:30};
  const matrix=rows.map(r=>fields.map(f=>{
    if(f==='drawing'||f==='human_note')return '';
    if(f==='human_decision')return '未判断';
    if(f==='auto_status')return labels[r[f]];
    const v=r[f]??'';
    return typeof v==='string'&&v.startsWith('=')?"'"+v:v;
  }));
  sheet.getRange(`A7:N${end}`).values=matrix;
  sheet.getRange(`A7:N${end}`).format.rowHeightPx=315;
  sheet.getRange(`B7:B${end}`).format.wrapText=true;
  sheet.getRange(`D7:E${end}`).format.fill='#FFF2C6';
  sheet.getRange(`E7:G${end}`).format.wrapText=true;
  sheet.getRange(`I7:J${end}`).format.wrapText=true;
  sheet.getRange(`M7:N${end}`).format.wrapText=true;
  sheet.getRange(`D7:D${end}`).dataValidation={rule:{type:'list',values:['未判断','保留','排除','待判断']}};
  for(const decision of ['保留','排除','待判断']){
    const fill={'保留':'#E6F2ED','排除':'#F9E8E5','待判断':'#FFF1D8'}[decision];
    sheet.getRange(`D7:D${end}`).conditionalFormats.add('containsText',{text:decision,format:{fill}});
  }
  for(let i=0;i<rows.length;i++){
    const bytes=await fs.readFile(path.join(root,rows[i].png_path));
    sheet.images.add({dataUrl:'data:image/png;base64,'+bytes.toString('base64'),anchor:{from:{row:i+6,col:2,rowOffsetPx:5,colOffsetPx:5},extent:{widthPx:460,heightPx:293}}});
    images++;
  }
  sheet.getRange('H4').values=[['本页人工标记数']];
  sheet.getRange('I4').formulas=[[`=COUNTIFS(D7:D${end},"保留")+COUNTIFS(D7:D${end},"排除")+COUNTIFS(D7:D${end},"待判断")`]];
  sheet.getRange('J4').values=[[`共 ${rows.length} 个结构`]];
}
const main=sheets[0].sheet;
const mainEnd=sheets[0].rows.length+6,excludedEnd=sheets[1].rows.length+6;
main.getRange('A4:F4').values=[['初筛保留',null,'初筛排除',null,'待判断',null]];
main.getRange('B4').formulas=[[`=COUNTIFS(F7:F${mainEnd},"初筛保留")`]];
main.getRange('D4').formulas=[[`=COUNTA('排除项复核'!A7:A${excludedEnd})`]];
main.getRange('F4').formulas=[[`=COUNTIFS(F7:F${mainEnd},"待人工判断")`]];
for(const cell of ['B4','D4','F4'])main.getRange(cell).setNumberFormat('#,##0');
// Exercise a real input and restore it before saving.
main.getRange('D7').values=[['保留']];
if(main.getRange('I4').values[0][0]!==1)throw Error('Human review progress did not update');
main.getRange('D7').values=[['未判断']];
wb.recalculate();
const actual=[main.getRange('B4').values[0][0],main.getRange('D4').values[0][0],main.getRange('F4').values[0][0]];
const expected=['retain','exclude','review'].map(k=>payload.summary.counts[k]);
if(JSON.stringify(actual)!==JSON.stringify(expected))throw Error('Workbook totals do not match graph partition');
if(main.getRange('I4').values[0][0]!==0)throw Error('Temporary manual decision was not restored');
const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:10},maxChars:1200});
console.log((await wb.inspect({kind:'table',range:'看图审核!A4:F4',include:'values,formulas',tableMaxRows:1,tableMaxCols:6,maxChars:1600})).ndjson);
console.log(scan.ndjson);
for(const {sheet} of sheets){
  try{await fs.access(path.join(root,'authoring',sheet.name+'.png'));continue}catch{}
  const blob=await wb.render({sheetName:sheet.name,range:'A1:F9',scale:1,format:'png'});
  await fs.writeFile(path.join(root,'authoring',sheet.name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
const pendingPreview=await wb.render({sheetName:'看图审核',range:'A51:G54',scale:1,format:'png'});
await fs.writeFile(path.join(root,'authoring','待判断.png'),new Uint8Array(await pendingPreview.arrayBuffer()));
const glycanPreview=await wb.render({sheetName:'看图审核',range:'A40:C43',scale:1,format:'png'});
await fs.writeFile(path.join(root,'authoring','糖头基.png'),new Uint8Array(await glycanPreview.arrayBuffer()));
const book=await SpreadsheetFile.exportXlsx(wb);
await book.save(path.join(root,'headgroup_structure_review.xlsx'));
await fs.writeFile(path.join(root,'authoring/workbook_qa.json'),JSON.stringify({sheets:2,images,rows:groups.map(([name,rows])=>({name,rows:rows.length})),totals:actual,decision_change_recalculated:true,temporary_decision_restored:true,formula_scan:scan.ndjson},null,2));
console.log(JSON.stringify({saved:'headgroup_structure_review.xlsx',images,totals:actual}));
