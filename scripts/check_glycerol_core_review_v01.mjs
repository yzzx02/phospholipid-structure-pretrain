// Bounded offline checks of the real HTML script, including decision and CSV export.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=path.resolve('outputs/glycerol_core_collection_v01_20261023');
const page=await fs.readFile(path.join(root,'index.html'),'utf8');
const script=page.match(/<script>([\s\S]*)<\/script>/)[1];
const nodes=new Map();
const document={getElementById(id){if(!nodes.has(id))nodes.set(id,{value:id==='scope'?'passed':'',innerHTML:'',textContent:'',showModal(){},close(){}});return nodes.get(id)},createElement(){return {click(){}}}};
const storage=new Map();const localStorage={getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)};
let captured;const URL={createObjectURL(blob){captured=blob;return 'blob:qa'},revokeObjectURL(){}};
const app=new Function('document','localStorage','URL','Blob','setTimeout','alert',script+';return {data:DATA,getVisible:()=>visible};')(document,localStorage,URL,Blob,()=>0,message=>{throw Error(message)});
assert.equal(app.data.length,352);
assert.equal(app.getVisible().length,136);
nodes.get('scope').value='all';nodes.get('scope').onchange();assert.equal(app.getVisible().length,352);
nodes.get('search').value='DGTS';nodes.get('search').oninput();assert(app.getVisible().some(r=>r.display_name.includes('DGTS')));
nodes.get('search').value='';nodes.get('search').oninput();
const id=app.data[0].core_id;nodes.get('grid').onclick({target:{dataset:{id,v:'保留'}}});assert(nodes.get('status').textContent.includes('保留 1'));
nodes.get('export').onclick();const csv=await captured.text();const bytes=new Uint8Array(await captured.arrayBuffer());assert.deepEqual([...bytes.slice(0,3)],[239,187,191]);assert.equal(csv.split('\r\n').length,353);assert(csv.includes(id));
nodes.get('backup').onclick();const backup=JSON.parse(await captured.text());assert.equal(backup.decisions[id].decision,'保留');
nodes.get('grid').onclick({target:{dataset:{id,v:''}}});assert(nodes.get('status').textContent.includes('保留 0'));
for(const r of app.data){await fs.access(path.join(root,'svg',r.core_id+'.svg'));await fs.access(path.join(root,'png',r.core_id+'.png'));assert.equal(typeof r.core_smiles,'string')}
const qa={graphs:app.data.length,initial_visible:136,all_filter:true,search:true,decision_edit:true,csv_BOM_and_real_newlines:true,backup_and_undo:true,all_graph_images_exist:true};
await fs.writeFile(path.join(root,'authoring/html_qa.json'),JSON.stringify(qa,null,2));console.log(JSON.stringify(qa));
