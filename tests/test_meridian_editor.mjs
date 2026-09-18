import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

const project=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const source=fs.readFileSync(path.join(project,"web/meridian_editor.js"),"utf8");
let extension;
class Element {
    constructor(tag){this.tagName=tag;this.children=[];this.listeners={};this.style={};this.value="";}
    appendChild(child){this.children.push(child);child.parent=this;return child;}
    replaceChildren(){this.children=[];}
    addEventListener(name,fn){this.listeners[name]=fn;}
    setAttribute(){}
    remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);this.removed=true;}
    showModal(){this.open=true;}
    getContext(){return {fillRect(){},fillStyle:""};}
    getBoundingClientRect(){return {left:0,top:0,width:620,height:350};}
    click(){this.listeners.click?.();}
}
const body=new Element("body");
const context={app:{registerExtension(v){extension=v;}},console,Math,JSON,Number,Object,Array,Set,
    document:{body,createElement(tag){return new Element(tag);}},alert(message){throw Error(message);},
    URL:{createObjectURL(){return "blob:owned";},revokeObjectURL(){}},Blob:class{}};
vm.createContext(context);
vm.runInContext(source.replace(/^import .*\n/,"").replace(/export /g,""),context);
const p={schema:"t8.meridian.camera.v1",geometry_id:"geo",origin_frame:0,window_end:20,frames:5,
    space:"source_window_first_camera_x_right_y_down_z_forward",units:"fixed_pivot_depth",roll:0,mode:"authored",pivot:[0,0,1],
    camera_keys:[{t:0,pos:[0,0,0],look:[0,0,1],focal:1,ease:true},{t:4,pos:[.08,0,0],look:[0,0,1],focal:1,ease:false}],
    time_keys:[{t:0,src:0},{t:4,src:4}]};
const material={identity:"geo",start:0,end:20,kind:"video",plan:p,canvas:[864,1184],
    clouds:[{src:0,points:[[0,0,1],[.1,0,1]],colors:[[200,100,100],[100,200,100]],thumbnail:"data:image/jpeg;base64,AA"}],
    source_intrinsics:[[[450,0,256],[0,450,256],[0,0,1]]]};
context.material=material;context.plan=p;
vm.runInContext("validateDraft(plan,material)",context);
for(const [name,change] of [
    ["stale",v=>v.geometry_id="other"],["reverse",v=>v.time_keys[1].src=-1],
    ["floatkey",v=>v.time_keys[1].t=3.5],["NaN",v=>v.camera_keys[0].focal=NaN],
    ["coincident",v=>v.camera_keys[0].look=[0,0,0]]]){
    context.invalid=structuredClone(p);change(context.invalid);
    assert.throws(()=>vm.runInContext("validateDraft(invalid,material)",context),undefined,name);
}
class Node {
    constructor(){this.widgets=[{name:"camera_plan",value:""},{name:"frames",value:5}];this.size=[100,100];}
    addWidget(type,name,value,callback,options){this.widgets.push({type,name,value,callback,options});}
    setSize(size){this.size=size;}
    setDirtyCanvas(){this.dirty=true;}
    onExecuted(){this.originalExecuted=true;return 71;}
    onNodeCreated(){return 73;}
    onRemoved(){return 79;}
}
await extension.beforeRegisterNodeDef(Node,{name:"unrelated"});
const other=new Node();assert.equal(other.onNodeCreated(),73);assert.equal(other.widgets.length,2);
await extension.beforeRegisterNodeDef(Node,{name:"MiniMaxH3MeridianCameraEXPT8"});
const node=new Node();assert.equal(node.onNodeCreated(),73);assert.equal(node.widgets.length,5);
assert.equal(node.size[0],620);assert.equal(node.widgets[2].options.serialize,false);
assert.equal(node.onExecuted({meridian_editor:[material]}),71);assert(node.originalExecuted);
node.widgets[2].callback();
const dialog=body.children[0];assert(dialog.open);
function all(el){return [el,...el.children.flatMap(all)];}
const add=all(dialog).find(el=>el.textContent==="添加时间关键帧");add.click();
const saved=JSON.parse(node.widgets[0].value);
assert.equal(saved.time_keys.length,3);assert.equal(saved.camera_keys.length,2);
assert.equal(node.widgets[1].value,5);assert(node.dirty);
// A late backend answer must not replace the newer draft.
node.onExecuted({meridian_editor:[{...material,plan:p,marker:"stale"}]});
assert.equal(node._t8MeridianPayload.marker,undefined);
const latest={...material,plan:saved,marker:"latest"};
node.onExecuted({meridian_editor:[latest]});assert.equal(node._t8MeridianPayload.marker,"latest");
assert.equal(node.onRemoved(),79);assert(dialog.removed);assert.equal(body.children.length,0);
// Planning before geometry only writes actual preset widgets, never a fake canonical geometry.
const preflight=new Node();preflight.widgets.push({name:"preset",value:"slide"},{name:"strength",value:.08});preflight.onNodeCreated();
context.preflight=preflight;
vm.runInContext('setPresetDraft(preflight,{preset:"orbit",strength:.12,frames:90})',context);
assert.equal(preflight.widgets.find(w=>w.name==="preset").value,"orbit");
assert.equal(preflight.widgets.find(w=>w.name==="frames").value,90);
assert.equal(preflight.widgets.find(w=>w.name==="camera_plan").value,"");assert.equal(preflight._t8MeridianPayload,null);
assert.throws(()=>vm.runInContext('setPresetDraft(preflight,{preset:"orbit",strength:.12,frames:89})',context));
preflight.widgets.find(w=>w.name==="打开大号运镜／时间编辑器").callback();
const planning=body.children[0];assert(planning.open);
assert(all(planning).some(el=>el.textContent?.includes("这里没有真实几何")));
all(planning).find(el=>el.textContent==="关闭，不应用").click();assert(planning.removed);
vm.runInContext('setPresetDraft(preflight,{preset:"slide",strength:.12,frames:73})',context);
preflight.widgets.find(w=>w.name==="打开大号运镜／时间编辑器").callback();
const slidePlanning=body.children[0];assert(slidePlanning.open);
assert(all(slidePlanning).some(el=>el.textContent?.includes("相机平行横移")&&el.textContent?.includes("朝向保持不变，不锁定目标")));
all(slidePlanning).find(el=>el.textContent==="关闭，不应用").click();assert(slidePlanning.removed);
assert(!source.includes("fetch("));assert(!source.includes("setInterval("));assert(!source.includes("http"));
console.log(JSON.stringify({status:"actual_source_fake_DOM_editor_pass_not_browser",cases:23,ordinaryQueueStarted:false}));
