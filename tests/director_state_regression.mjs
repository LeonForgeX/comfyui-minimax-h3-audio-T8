// State-sequence regressions. No browser, network, Core, model or GPU is used.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { setTimeout as realDelay } from 'node:timers/promises';
import { makeDirectorServices } from '../web/director/session.mjs';
import { createSamplingDialog } from '../web/director/sampling_ui.mjs';

const copy = structuredClone;
const storage = () => {
    const values = new Map();
    return { getItem:k=>values.get(k)??null, setItem:(k,v)=>values.set(k,String(v)), removeItem:k=>values.delete(k) };
};
const project = (id='project-a', revision=1, text='saved') => ({
    schema:'t8.minimax_h3.director_project', version:2, id, revision, title:'test',
    current:'shot-1', assets:[], doc:{shots:[{id:'shot-1',simplePrompt:text}]},
});
function serviceHarness({local=null, specific=null, pending=false, state='running', server=project()}={}) {
    globalThis.localStorage=storage(); globalThis.sessionStorage=storage();
    sessionStorage.setItem('t8director.tab','test');
    if(local) localStorage.setItem('t8director.draft:test',JSON.stringify(local));
    if(pending) sessionStorage.setItem('t8director.activeJob:test',JSON.stringify({prompt_id:'original',shot_id:'shot-1'}));
    globalThis.location={href:'http://test.invalid/ui'+(specific?'?project_id='+specific:''),origin:'http://test.invalid'};
    const messages=[];
    globalThis.window={addEventListener(){},history:{state:null,replaceState(_state,_unused,url){location.href=String(url)}}}; globalThis.parent={postMessage:value=>messages.push(value)};
    globalThis.MutationObserver=class{observe(){}};
    const timers=[],intervals=new Map();let timerId=0;
    globalThis.setTimeout=fn=>{timers.push(fn);return 1};
    globalThis.setInterval=fn=>{intervals.set(++timerId,fn);return timerId};
    globalThis.clearInterval=id=>intervals.delete(id);
    const node=()=>({dataset:{},click(){},textContent:'',hidden:true,open:false,insertAdjacentHTML(){},append(){},querySelector(){return null},close(){this.open=false},addEventListener(){}});
    globalThis.document={createElement:node};
    const appended=[],nodes=new Map(),listeners={},requests=[],notices=[],dialogs=[],records=[],results=[];
    const $=selector=>{if(!nodes.has(selector))nodes.set(selector,node());return nodes.get(selector)};
    const root={...node(),append:n=>appended.push(n),addEventListener:(event,fn)=>listeners[event]=fn};
    let doc=project().doc,current='shot-1',jobState=state,assets=new Map();
    globalThis.fetch=async(url,opts)=>{
        const path=new URL(url).pathname; requests.push({path,method:opts.method,body:opts.body&&JSON.parse(opts.body)});
        let value={};
        if(path.includes('/projects/'))value=opts.method==='POST'?{revision:2}:server;
        else if(path.endsWith('/cancel'))value={deleted_from_queue:true};
        else if(path.endsWith('/jobs/original'))value={state:jobState,outputs:{}};
        else if(path.endsWith('/generate'))value={prompt_id:'second',recipe:'test'};
        else if(path.endsWith('/jobs/second'))value={state:'success',outputs:{}};
        else if(path.endsWith('/compile'))value={ready:true};
        else if(path.includes('/results/'))value={results:[]};
        return {ok:true,json:async()=>value};
    };
    const showDialog=(...args)=>{dialogs.push(args);const dialog=$('[data-dialog]');delete dialog.dataset.jobId;dialog.open=true;};
    const service=makeDirectorServices({root,$,esc:String,notify:t=>notices.push(t),showDialog,tokenMap:()=>new Map(),checkpoint(){},doc:()=>doc,assets:()=>assets,current:()=>current,replace:(d,c,a)=>{doc=d;current=c;assets=a},setResults:value=>results.push(value),recordResult:value=>records.push(value),resetHistory(){},render(){}});
    const click=dataset=>listeners.click({target:{closest:()=>({dataset,disabled:false})},preventDefault(){},stopImmediatePropagation(){}});
    return {appended,assets:()=>assets,input:target=>listeners.input({target}),service,requests,notices,dialogs,timers,intervals,click,$,records,results,messages,doc:()=>doc,setCurrent:value=>{current=value},setState:value=>{jobState=value}};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const waitFor=async predicate=>{const deadline=Date.now()+2000;while(!predicate()&&Date.now()<deadline)await realDelay(1);assert.ok(predicate(),'Expected asynchronous test state before timeout');};
const response=(data,ok=true,status=200)=>({ok,status,json:async()=>data});

for(const change of ['shot','project','draft','new-request','none'])test(`compile response context guard: ${change}`,async()=>{
    const p=project();p.doc.shots.push({id:'shot-2',simplePrompt:'second'});
    const h=serviceHarness({local:p,specific:p.id,server:project('project-b')});await h.service.restore();
    const prior=globalThis.fetch,waiting=[];
    globalThis.fetch=(url,opts)=>new URL(url).pathname.endsWith('/compile')?new Promise(resolve=>waiting.push(resolve)):prior(url,opts);
    const data={ready:true,shots:[{id:'shot-1',canvas:{preprocessing:[]}}],errors:[],warnings:[]};
    const checking=h.service.compile();await flush();
    if(change==='shot')h.setCurrent('shot-2');
    if(change==='project'){globalThis.confirm=()=>true;await h.click({openProject:'project-b'});}
    if(change==='draft'){h.doc().shots[0].simplePrompt='changed';h.service.draft();}
    if(change==='new-request'){const newer=h.service.compile();await flush();waiting[1](response(data));await newer;}
    waiting[0](response(data));await checking;
    assert.equal(h.dialogs.length,['none','new-request'].includes(change)?1:0);
});

for(const status of [404,500])test(`late old poll ${status} cannot affect a new active job`,async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});await h.service.restore();
    sessionStorage.setItem('t8director.activeJob:test',JSON.stringify({prompt_id:'original',project_id:'project-a',shot_id:'shot-1'}));
    await h.service.restore();const watching=h.timers.shift()();await flush();
    const prior=globalThis.fetch;let finishOld;let secondState='running';
    globalThis.fetch=(url,opts)=>{
        const path=new URL(url).pathname;
        if(path.endsWith('/jobs/original'))return new Promise(resolve=>finishOld=()=>resolve(response({error:'old failed'},false,status)));
        if(path.endsWith('/jobs/second'))return Promise.resolve(response({state:secondState,outputs:{}}));
        return prior(url,opts);
    };
    const polling=[...h.intervals.values()][0]();await flush();
    await h.click({service:'cancel-job'});await watching;
    const generating=h.click({action:'generate'});await waitFor(()=>h.records.length&&h.intervals.size);
    const notices=h.notices.length;finishOld();await polling;assert.equal(h.notices.length,notices);
    assert.equal(JSON.parse(sessionStorage.getItem('t8director.activeJob:test')).prompt_id,'second');
    secondState='success';await [...h.intervals.values()][0]();await generating;
    assert.equal(h.records.at(-1).state,'success');assert.equal(h.intervals.size,0);
    await h.click({action:'generate'});assert.equal(h.requests.filter(x=>x.path.endsWith('/generate')).length,2);
});

for(const freshEmpty of [false,true])test(`old result query cannot erase completed video; fresh index empty=${freshEmpty}`,async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'}),prior=globalThis.fetch;
    let finishInitial,count=0;
    const done={shot_id:'shot-1',prompt_id:'second',state:'success',outputs:{save:{videos:[{filename:'new.mp4'}]}}};
    globalThis.fetch=(url,opts)=>{
        const path=new URL(url).pathname;
        if(path.includes('/results/'))return ++count===1?new Promise(resolve=>finishInitial=()=>resolve(response({results:[]}))):Promise.resolve(response({results:freshEmpty?[]:[done]}));
        if(path.endsWith('/jobs/second'))return Promise.resolve(response(done));
        return prior(url,opts);
    };
    await h.service.restore();await h.click({action:'generate'});await flush();
    assert.equal(h.results.at(-1)[0].prompt_id,'second');
    finishInitial();await flush();assert.equal(h.results.at(-1)[0].prompt_id,'second');
    // Loading a different project must not retain the previous project's completions.
    globalThis.confirm=()=>true;globalThis.fetch=(url,opts)=>new URL(url).pathname.includes('/projects/')?Promise.resolve(response(project('project-b'))):new URL(url).pathname.includes('/results/')?Promise.resolve(response({results:[]})):prior(url,opts);
    await h.click({openProject:'project-b'});await flush();assert.deepEqual(h.results.at(-1),[]);
});

for(const change of ['project','round-trip','none','picker-stale','shot'])test(`upload binds only original project context: ${change}`,async()=>{
    const html=readFileSync(new URL('../web/director/index.html',import.meta.url),'utf8');
    const source=html.slice(html.indexOf('async function importFiles('),html.indexOf("$('[data-picker]').addEventListener"));
    let finish,token='A:1',calls=0;
    const a={shots:[{id:'A-shot',rev:1},{id:'other',rev:1}],sharedRefs:[]},b={shots:[{id:'B-shot',rev:1}],sharedRefs:[]};
    const notices=[],state={doc:a,assets:new Map(),view:'output',pair:false,services:{contextToken:()=>token},notify:t=>notices.push(t),readAsset:()=>{calls++;return new Promise(resolve=>finish=()=>resolve({id:'asset-'+calls,kind:'image'}));},referenceSnapshot:()=>new Map(),checkpoint(){},reconcileTokens(){},persist(){},render(){}};
    state.sh=()=>state.doc.shots[0];vm.createContext(state);vm.runInContext(source,state);
    if(change==='picker-stale')token='B:2';
    const upload=vm.runInContext("importFiles([{},{}],{target:'global',shot:'A-shot',context:'A:1'})",state);
    if(change==='picker-stale'){await upload;assert.equal(calls,0);return;}
    if(change==='project'){state.doc=b;state.assets=new Map();token='B:2';}
    if(change==='round-trip')token='A:3';
    if(change==='shot')state.sh=()=>a.shots[1];
    finish();await flush();
    if(['none','shot'].includes(change)){assert.equal(calls,2);finish();}
    await upload;
    if(['none','shot'].includes(change)){assert.deepEqual(a.sharedRefs,['asset-1','asset-2']);assert.equal(state.assets.size,2);}
    else {assert.equal(calls,1);assert.deepEqual(a.sharedRefs,[]);assert.deepEqual(b.sharedRefs,[]);assert.equal(state.assets.size,0);assert.match(notices.at(-1),/未绑定/);}
});

test('insertion label and insertion share the same target across shots and writing modes',()=>{
    const html=readFileSync(new URL('../web/director/index.html',import.meta.url),'utf8');
    const source=html.slice(html.indexOf('function resolveInsertTarget('),html.indexOf('\n',html.indexOf('function resolveInsertTarget(')));
    const s={id:'two',writingMode:'simple',events:[{id:'event-2'}]},state={lastTextTarget:{kind:'global',start:4},sh:()=>s,targetValue:t=>t.kind==='global'?'global':t.kind==='event'?'event text':'local'};
    vm.createContext(state);vm.runInContext(source,state);
    assert.equal(vm.runInContext('resolveInsertTarget().kind',state),'global');
    state.lastTextTarget={kind:'simplePrompt',shot:'one',start:1};
    assert.equal(vm.runInContext('resolveInsertTarget().shot',state),'two');
    s.writingMode='advanced';assert.equal(vm.runInContext('resolveInsertTarget().kind',state),'event');
    state.lastTextTarget={kind:'simplePrompt',shot:'two',start:1};assert.equal(vm.runInContext('resolveInsertTarget().kind',state),'event');
    const render=html.slice(html.indexOf('function renderWriting('),html.indexOf('function resolveInsertTarget('));
    const insert=html.slice(html.indexOf('function insertAsset('),html.indexOf('function insertEditorAsset('));
    assert.match(render,/targetLabel\(t\)/);assert.match(render,/resolveInsertTarget\(\)/);assert.match(insert,/resolveInsertTarget\(\)/);
});
const localDraft=()=>JSON.parse(localStorage.getItem('t8director.draft:test'));
function delaySave() {
    const fetchNow=globalThis.fetch;
    let finish;
    globalThis.fetch=(url,options)=>options.method==='POST'&&new URL(url).pathname.includes('/projects/')
        ?new Promise(resolve=>{finish=(ok=true)=>resolve({ok,status:ok?200:409,json:async()=>ok?{revision:2}:{error:'revision conflict'}})})
        :fetchNow(url,options);
    return ok=>finish(ok);
}

for(const asCopy of [false,true])test(`save preserves typing during response and reload, copy=${asCopy}`,async()=>{
    const h=serviceHarness({local:project('project-a',1,'before-save'),specific:'project-a'});
    await h.service.restore();
    const finish=delaySave(),saving=h.click({service:asCopy?'copy':'save'});
    await flush();h.doc().shots[0].simplePrompt='typed-during-save';h.service.draft();
    finish();await saving;
    assert.equal(localDraft().doc.shots[0].simplePrompt,'typed-during-save');
    assert.equal(localDraft().revision,2);
    assert.match(h.$('[data-save]').textContent,/后续编辑/);
    assert.equal(h.messages.at(-1).project.doc.shots[0].simplePrompt,'before-save'); // actual server snapshot only
    const savedId=h.service.envelope().id;
    assert.equal(new URL(location.href).searchParams.get('project_id'),savedId);
    if(asCopy)assert.notEqual(savedId,'project-a');
    await h.service.restore();assert.equal(h.service.envelope().doc.shots[0].simplePrompt,'typed-during-save');
    assert.equal(h.service.envelope().id,savedId);
});
test('failed copy leaves ID, revision, URL and new draft unchanged',async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});await h.service.restore();
    const finish=delaySave(),saving=h.click({service:'copy'});await flush();
    h.doc().shots[0].simplePrompt='keep after failure';h.service.draft();finish(false);await saving;
    assert.equal(h.service.envelope().id,'project-a');assert.equal(h.service.envelope().revision,1);
    assert.equal(new URL(location.href).searchParams.get('project_id'),'project-a');
    assert.equal(localDraft().doc.shots[0].simplePrompt,'keep after failure');
    assert.equal(h.messages.some(m=>m.type==='t8-director:saved'),false);
});
test('save response after opening another project cannot replace its identity or draft',async()=>{
    const h=serviceHarness({local:project(),specific:'project-a',server:project('project-b',7,'B')});await h.service.restore();
    const finish=delaySave(),saving=h.click({service:'save'});await flush();
    globalThis.confirm=()=>true;await h.click({openProject:'project-b'});
    finish();await saving;
    assert.equal(h.service.envelope().id,'project-b');assert.equal(localDraft().id,'project-b');
    assert.equal(localDraft().revision,7);assert.equal(localDraft().doc.shots[0].simplePrompt,'B');
    assert.equal(new URL(location.href).searchParams.get('project_id'),'project-b');
    assert.equal(h.messages.some(m=>m.type==='t8-director:saved'),false);
});
test('successful unedited save reports saved and keeps the CAS revision',async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});await h.service.restore();await h.click({service:'save'});
    assert.equal(localDraft().revision,2);assert.match(h.$('[data-save]').textContent,/已真实保存/);
    assert.equal(h.requests.find(r=>r.method==='POST'&&r.path.includes('/projects/')).body.expected_revision,1);
});
for(const owner of ['project-a','project-b',undefined])test(`restored task retains owner ${owner} without assigning foreign results`,async()=>{
    const h=serviceHarness({local:project('project-b'),specific:'project-b'});await h.service.restore();
    sessionStorage.setItem('t8director.activeJob:test',JSON.stringify({prompt_id:'original',project_id:owner,shot_id:'shot-1'}));
    await h.service.restore();const watching=h.timers.shift()();await flush();
    assert.equal(JSON.parse(sessionStorage.getItem('t8director.activeJob:test')).project_id,owner);
    h.setState('success');await [...h.intervals.values()][0]();await watching;
    assert.equal(h.records.length,owner==='project-b'?1:0);
    if(h.records.length)assert.equal(h.records[0].project_id,owner);
    assert.equal(sessionStorage.getItem('t8director.activeJob:test'),null);
});
test('running task stays with original project after a project switch with identical shot IDs',async()=>{
    const h=serviceHarness({local:project(),specific:'project-a',server:project('project-b')});await h.service.restore();
    sessionStorage.setItem('t8director.activeJob:test',JSON.stringify({prompt_id:'original',project_id:'project-a',shot_id:'shot-1'}));
    await h.service.restore();const watching=h.timers.shift()();await flush();
    globalThis.confirm=()=>true;await h.click({openProject:'project-b'});
    h.setState('success');await [...h.intervals.values()][0]();await watching;
    assert.equal(h.records.length,0);assert.equal(h.service.envelope().id,'project-b');
});
test('task whose shot was deleted does not reinsert a result into current project',async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});await h.service.restore();
    sessionStorage.setItem('t8director.activeJob:test',JSON.stringify({prompt_id:'original',project_id:'project-a',shot_id:'deleted'}));
    await h.service.restore();const watching=h.timers.shift()();await flush();
    h.setState('success');await [...h.intervals.values()][0]();await watching;assert.equal(h.records.length,0);
});
for(const action of ['generate','generate-all'])test(`${action} submission response cannot acquire newly opened project identity`,async()=>{
    const h=serviceHarness({local:project(),specific:'project-a',server:project('project-b')});await h.service.restore();
    const fetchNow=globalThis.fetch;let finishSubmit;
    globalThis.fetch=(url,options)=>new URL(url).pathname.endsWith('/generate')
        ?new Promise(resolve=>{finishSubmit=()=>resolve({ok:true,json:async()=>({prompt_id:'original',recipe:'test'})})})
        :fetchNow(url,options);
    const generating=h.click({action});
    await waitFor(()=>finishSubmit);
    assert.ok(finishSubmit);globalThis.confirm=()=>true;await h.click({openProject:'project-b'});
    finishSubmit();await flush();
    assert.equal(JSON.parse(sessionStorage.getItem('t8director.activeJob:test')).project_id,'project-a');
    h.setState('success');await [...h.intervals.values()][0]();await generating;
    assert.equal(h.records.length,0);assert.equal(h.requests.filter(r=>r.path.endsWith('/compile')).length,action==='generate-all'?1:0);
});

test('matching project URL restores local draft and original revision, even with newer server',async()=>{
    const h=serviceHarness({local:project('project-a',1,'unsaved'),specific:'project-a',server:project('project-a',2,'new server')});
    await h.service.restore();
    assert.equal(h.service.envelope().doc.shots[0].simplePrompt,'unsaved');
    assert.equal(h.service.envelope().revision,1); // CAS must not silently advance.
    assert.equal(h.requests.some(r=>r.path.includes('/projects/')),false);
    assert.equal(JSON.parse(localStorage.getItem('t8director.draft:test')).doc.shots[0].simplePrompt,'unsaved');
});
test('different explicit project loads requested project and backs up previous tab draft',async()=>{
    const original=project('project-a',1,'keep me');
    const h=serviceHarness({local:original,specific:'project-b',server:project('project-b',3)});
    await h.service.restore();
    assert.equal(h.service.envelope().id,'project-b');
    assert.deepEqual(JSON.parse(localStorage.getItem('t8director.draft:test:backup:project-a')),original);
});
test('ordinary no-ID reload still restores local draft',async()=>{
    const h=serviceHarness({local:project('project-a',1,'local')});await h.service.restore();
    assert.equal(h.service.envelope().doc.shots[0].simplePrompt,'local');
});
test('corrupt local draft bytes are backed up before loading an explicit project',async()=>{
    const h=serviceHarness({specific:'project-a'});localStorage.setItem('t8director.draft:test','{bad');
    await h.service.restore();
    assert.equal(localStorage.getItem('t8director.draft:test:backup:unreadable'),'{bad');
    assert.equal(h.service.envelope().id,'project-a');
});
test('pending restore blocks single and batch generation before and during polling',async()=>{
    const h=serviceHarness({pending:true});await h.service.restore();
    await h.click({action:'generate'});await h.click({action:'generate-all'});
    const watching=h.timers.shift()();await flush();
    await h.click({action:'generate'});await h.click({action:'generate-all'});
    assert.equal(h.requests.some(r=>r.path.endsWith('/generate')||r.path.endsWith('/compile')),false);
    assert.equal(h.intervals.size,1);
    assert.equal(h.$('[data-service="job-status"]').hidden,false);
    await h.click({service:'job-status'});assert.match(h.dialogs.at(-1)[1],/original/);
    await h.click({service:'cancel-job'});await watching;
    assert.equal(h.intervals.size,0);assert.equal(h.$('[data-service="job-status"]').hidden,true);
    await h.click({action:'generate'});
    assert.equal(h.requests.filter(r=>r.path.endsWith('/generate')).length,1);
});
for(const state of ['success','error'])test(`restored ${state} task unlocks generation`,async()=>{
    const h=serviceHarness({pending:true,state});await h.service.restore();await h.timers.shift()();
    assert.equal(h.intervals.size,0);
    await h.click({action:'generate'});assert.equal(h.requests.filter(r=>r.path.endsWith('/generate')).length,1);
});

function samplingHarness(catalog={}) {
    const body={innerHTML:'',insertAdjacentHTML(_position,html){this.innerHTML=html+this.innerHTML}};
    const listeners={},notices=[],applied=[];
    const dialog={open:false,querySelector:s=>s==='[data-model-settings]'?body:{focus(){},setAttribute(){},setSelectionRange(){},insertAdjacentHTML:(where,html)=>body.insertAdjacentHTML(where,html)},addEventListener:(key,fn)=>listeners[key]=fn,showModal(){this.open=true},close(){this.open=false}};
    const data={generation:{resolution_mp:'auto'},sampling:{mode:'single'},shots:[{samplingInherit:true}]};
    const api=createSamplingDialog({root:{querySelector:()=>dialog},doc:()=>data,shot:()=>data.shots[0],catalog:()=>catalog,notify:t=>notices.push(t),apply:value=>applied.push(value)});
    const click=dataset=>listeners.click({target:{closest:selector=>selector==='button'?{dataset,closest:()=>null}:null}});
    const mp=value=>listeners.change({target:{dataset:{},value,hasAttribute:key=>key==='data-sampling-mp'}});
    const enabled=key=>listeners.change({target:{dataset:{samplingField:'enabled'},checked:true,hasAttribute:()=>false,closest:selector=>({dataset:selector==='[data-stage]'?{stage:key}:{index:'0'}})}});
    const rowField=(key,field,value,index=0)=>listeners.change({target:{dataset:{samplingField:field},value,checked:value,hasAttribute:()=>false,closest:selector=>({dataset:selector==='[data-stage]'?{stage:key}:{index:String(index)}})}});
    const search=(key,value)=>listeners.input({target:{dataset:{samplingSearch:key},value,selectionStart:value.length}});
    return {api,body,dialog,notices,applied,click,mp,enabled,data,rowField,search};
}
test('local/global/local round-trip preserves MP draft and cancels transaction cleanly',()=>{
    const h=samplingHarness();h.api.open();h.click({samplingScope:'local'});h.mp('0.5');
    h.click({samplingScope:'global'});h.click({samplingScope:'local'});h.api.commit();
    assert.equal(h.applied[0].local.resolution_mp,'0.5');assert.equal(h.applied[0].inherited,false);
    h.api.open();h.click({samplingScope:'local'});h.mp('0.8');h.api.close();assert.equal(h.applied.length,1);
});
test('inactive two-pass empty enabled LoRA remains a draft and does not block single',()=>{
    const h=samplingHarness();h.api.open();h.click({samplingMode:'two_pass'});h.click({samplingAction:'add',stage:'low_loras'});h.enabled('low_loras');
    h.click({samplingMode:'single'});h.api.commit();
    assert.equal(h.applied.length,1);assert.equal(h.applied[0].global.mode,'single');
    assert.equal(h.applied[0].global.two_pass.low_loras[0].enabled,true);
});
test('active empty LoRA is still rejected with stage and row shown inside dialog',()=>{
    const h=samplingHarness();h.api.open();h.click({samplingMode:'two_pass'});h.click({samplingAction:'add',stage:'high_loras'});h.enabled('high_loras');h.api.commit();
    assert.equal(h.applied.length,0);assert.equal(h.dialog.open,true);assert.match(h.body.innerHTML,/role="alert"/);assert.match(h.body.innerHTML,/二采第 1 条/);
});
test('inactive local scope does not block applying valid global settings',()=>{
    const h=samplingHarness();h.api.open();h.click({samplingScope:'local'});h.click({samplingMode:'two_pass'});h.click({samplingAction:'add',stage:'low_loras'});h.enabled('low_loras');h.click({samplingScope:'global'});h.api.commit();
    assert.equal(h.applied.length,1);assert.equal(h.applied[0].inherited,true);
});
test('error navigation and correction preserves requested independent single sampling',()=>{
    const h=samplingHarness();h.api.open();h.click({samplingMode:'two_pass'});h.click({samplingAction:'add',stage:'low_loras'});h.enabled('low_loras');
    h.click({samplingScope:'local'});h.click({samplingMode:'single'});h.mp('0.5');h.api.commit();
    assert.equal(h.applied.length,0);assert.match(h.body.innerHTML,/全片默认.*已启用的 LoRA/s);
    h.rowField('low_loras','name','valid.safetensors');h.api.commit();
    assert.equal(h.applied.length,1);assert.equal(h.applied[0].inherited,false);
    assert.equal(h.applied[0].global.mode,'two_pass');assert.equal(h.applied[0].local.mode,'single');assert.equal(h.applied[0].local.resolution_mp,'0.5');
});
test('explicit scope choice after error can still switch back to global',()=>{
    const h=samplingHarness();h.api.open();h.click({samplingAction:'add',stage:'loras'});h.enabled('loras');
    h.click({samplingScope:'local'});h.api.commit();h.rowField('loras','enabled',false);
    h.click({samplingScope:'global'});h.api.commit();assert.equal(h.applied[0].inherited,true);
});
for(const [mode,stage] of [['single','loras'],['two_pass','low_loras'],['two_pass','high_loras']]) {
    for(const strength of ['99','-2.01','','NaN','Infinity'])test(`${stage} rejects invalid strength ${JSON.stringify(strength)} even disabled`,()=>{
        const h=samplingHarness();h.api.open();h.click({samplingMode:mode});h.click({samplingAction:'add',stage});h.rowField(stage,'strength',strength);h.api.commit();
        assert.equal(h.applied.length,0);assert.equal(h.dialog.open,true);assert.match(h.body.innerHTML,/强度必须是 -2–2/);
        h.rowField(stage,'strength','0.35');h.api.commit();assert.equal(h.applied[0].global[stage][0].strength,0.35);
    });
}
for(const strength of ['-2','0','2'])test(`valid LoRA boundary ${strength} retained`,()=>{
    const h=samplingHarness();h.api.open();h.click({samplingAction:'add',stage:'loras'});h.rowField('loras','strength',strength);h.api.commit();assert.equal(h.applied[0].global.loras[0].strength,Number(strength));
});
test('search preserves installed selected label but still marks actually missing file',()=>{
    const h=samplingHarness({lora:[{value:'known.safetensors',label:'Installed LoRA'}]});h.api.open();h.click({samplingAction:'add',stage:'loras'});
    h.rowField('loras','name','known.safetensors');h.search('loras','zz-no-match');
    assert.match(h.body.innerHTML,/<option value="known.safetensors" selected>Installed LoRA<\/option>/);assert.doesNotMatch(h.body.innerHTML,/本机未找到/);
    h.rowField('loras','name','missing.safetensors');assert.match(h.body.innerHTML,/missing.safetensors（本机未找到）/);
});
test('redo deletion repairs actual current ID before persistence, including undo redo cycles',()=>{
    const source=readFileSync(new URL('../web/director/index.html',import.meta.url),'utf8');
    const helper=source.slice(source.indexOf('function restoreHistoryState('),source.indexOf('\n',source.indexOf('function restoreHistoryState(')));
    const actions=source.slice(source.indexOf("case 'undo':"),source.indexOf("case 'new':"));
    const snapshots=[];
    const state={doc:{shots:[{id:'b'}]},current:'b',history:[{shots:[{id:'a'},{id:'b'}]}],future:[],clone:structuredClone,notify(){},render(){}};
    state.persist=()=>snapshots.push({doc:structuredClone(state.doc),current:state.current});vm.createContext(state);vm.runInContext(helper,state);
    for(let i=0;i<3;i++) {
        vm.runInContext("switch('undo'){"+actions+'}',state);state.current='a';
        vm.runInContext("switch('redo'){"+actions+'}',state);assert.equal(state.current,'b');
    }
    assert.ok(snapshots.every(p=>p.doc.shots.some(s=>s.id===p.current)));
});
test('result and output views reveal the actual stage and synchronize toggle state',()=>{
    const source=readFileSync(new URL('../web/director/index.html',import.meta.url),'utf8');
    const fn=source.slice(source.indexOf('function revealResultPreview()'),source.indexOf('const workbenchStage='));
    for(const view of ['input','output','result']) {
        const stage={hidden:true},toggle={textContent:'展开预览',setAttribute(k,v){this[k]=v}},root={dataset:{previewCollapsed:'true'}};
        vm.runInNewContext(fn+';revealResultPreview();',{view,root,$:s=>s==='[data-stage]'?stage:toggle});
        assert.equal(stage.hidden,view==='input');
        if(view!=='input'){assert.equal(toggle['aria-expanded'],'true');assert.equal(root.dataset.previewCollapsed,'false');}
    }
    assert.match(source,/renderStage=function\(\)\{revealResultPreview\(\)/);
});

test('late cancel response cannot stop the next task or unlock generation',async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});await h.service.restore();
    sessionStorage.setItem('t8director.activeJob:test',JSON.stringify({prompt_id:'original',project_id:'project-a',shot_id:'shot-1'}));
    await h.service.restore();const watching=h.timers.shift()();await flush();
    const prior=fetch;let finishCancel,secondState='running';
    globalThis.fetch=(url,opts)=>{
        const path=new URL(url).pathname;
        if(path.endsWith('/cancel'))return new Promise(r=>finishCancel=()=>r(response({deleted_from_queue:true})));
        if(path.endsWith('/jobs/second'))return Promise.resolve(response({state:secondState,outputs:{}}));
        return prior(url,opts);
    };
    const cancelling=h.click({service:'cancel-job'});await flush();
    h.setState('unknown');await [...h.intervals.values()][0]();await watching;
    const generating=h.click({action:'generate'});await waitFor(()=>h.records.length&&h.intervals.size);
    const notices=h.notices.length;finishCancel();await cancelling;
    assert.equal(h.intervals.size,1);assert.equal(h.notices.length,notices);
    assert.equal(JSON.parse(sessionStorage.getItem('t8director.activeJob:test')).prompt_id,'second');
    await h.click({action:'generate'});assert.equal(h.requests.filter(r=>r.path.endsWith('/generate')).length,1);
    secondState='success';await [...h.intervals.values()][0]();await generating;
    assert.equal(h.intervals.size,0);
});

for(const change of ['project','round-trip','picker-stale','target','none'])test(`reconnect retains initiating project and asset: ${change}`,async()=>{
    const a=project(),b=project('project-b');
    for(const p of [a,b]){p.assets=[{id:'old-image',kind:'image',name:'old'},{id:'other-image',kind:'image',name:'other'}];p.doc.sharedRefs=['old-image'];Object.assign(p.doc.shots[0],{first:'old-image',refs:[],tray:['old-image'],rev:1});}
    const h=serviceHarness({local:a,specific:a.id,server:b});await h.service.restore();
    let xhr,uploads=0;globalThis.XMLHttpRequest=class{constructor(){uploads++;xhr=this;this.upload={};}open(){}send(){}};
    globalThis.confirm=()=>true;await h.click({reconnectAsset:'old-image'});
    if(change==='picker-stale')await h.click({openProject:b.id});
    const picker=h.appended[1];picker.files=[new Blob(['x'],{type:'image/png'})];
    const uploading=picker.onchange();await flush();
    if(change==='picker-stale'){await uploading;assert.equal(uploads,0);return;}
    if(['project','round-trip'].includes(change))await h.click({openProject:b.id});
    if(change==='round-trip'){const prior=fetch;globalThis.fetch=(url,opts)=>new URL(url).pathname.endsWith('/projects/project-a')?Promise.resolve(response(a)):prior(url,opts);await h.click({openProject:a.id});}
    if(change==='target')await h.click({reconnectAsset:'other-image'});
    xhr.status=200;xhr.responseText=JSON.stringify({id:'new-image',kind:'image',name:'replacement'});xhr.onload();xhr.onloadend();await uploading;
    assert.equal(h.doc().shots[0].first,change==='none'?'new-image':'old-image');
    assert.deepEqual(h.doc().sharedRefs,[change==='none'?'new-image':'old-image']);
    assert.equal(h.assets().has('new-image'),change==='none');
    if(change!=='none')assert.match(h.notices.at(-1),/未绑定/);
});

for(const order of ['older-first','newer-first'])test(`only latest project open intent applies: ${order}`,async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});await h.service.restore();
    const prior=fetch,waiting=new Map();globalThis.confirm=()=>true;
    globalThis.fetch=(url,opts)=>{const path=new URL(url).pathname;return path.includes('/projects/')?new Promise(r=>waiting.set(path.split('/').at(-1),r)):prior(url,opts);};
    const older=h.click({openProject:'project-b'}),newer=h.click({openProject:'project-c'});await flush();
    if(order==='older-first'){waiting.get('project-b')(response(project('project-b')));await older;assert.equal(h.service.envelope().id,'project-a');}
    waiting.get('project-c')(response(project('project-c')));await newer;
    h.doc().shots[0].simplePrompt='new C edit';h.service.draft();
    if(order==='newer-first'){waiting.get('project-b')(response(project('project-b')));await older;}
    assert.equal(h.service.envelope().id,'project-c');assert.equal(localDraft().doc.shots[0].simplePrompt,'new C edit');
    assert.equal(JSON.parse(localStorage.getItem('t8director.draft:test:backup:project-a')).id,'project-a');
});

for(const edit of ['document','modal','none'])test(`opening project preserves edits during load: ${edit}`,async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});await h.service.restore();
    const prior=fetch;let finish;globalThis.confirm=()=>true;
    globalThis.fetch=(url,opts)=>new URL(url).pathname.includes('/projects/')?new Promise(r=>finish=()=>r(response(project('project-b')))):prior(url,opts);
    const opening=h.click({openProject:'project-b'});await flush();
    if(edit==='document'){h.doc().shots[0].simplePrompt='keep new text';h.service.draft();}
    if(edit==='modal')h.input({matches:()=>false});
    finish();await opening;assert.equal(h.service.envelope().id,edit==='none'?'project-b':'project-a');
    if(edit!=='none')assert.match(h.notices.at(-1),/新编辑/);
});

test('late JSON import cannot replace a subsequently opened project',async()=>{
    const h=serviceHarness({local:project(),specific:'project-a',server:project('project-c')});await h.service.restore();
    const prior=fetch;let finish;globalThis.confirm=()=>true;
    globalThis.fetch=(url,opts)=>new URL(url).pathname.endsWith('/validate')?new Promise(r=>finish=()=>r(response({project:project('project-b')}))):prior(url,opts);
    h.appended[0].files=[new Blob([JSON.stringify(project('project-b'))])];
    const importing=h.appended[0].onchange();await waitFor(()=>finish);
    await h.click({openProject:'project-c'});finish();await importing;
    assert.equal(h.service.envelope().id,'project-c');assert.equal(localDraft().id,'project-c');
});

for(const change of ['shot','project','draft','new-request','none'])test(`D3 preflight ignores stale context: ${change}`,async()=>{
    const p=project();p.doc.shots.push({id:'shot-2',simplePrompt:'other'});
    const h=serviceHarness({local:p,specific:p.id,server:project('project-b')});await h.service.restore();
    const prior=fetch,waiting=[];globalThis.confirm=()=>true;
    globalThis.fetch=(url,opts)=>new URL(url).pathname.endsWith('/d3/preflight')?new Promise(r=>waiting.push(r)):prior(url,opts);
    const data={capabilities:[],compile:{shot_id:'shot-1'}};
    const checking=h.click({service:'d3-preflight'});await flush();
    if(change==='shot')h.setCurrent('shot-2');
    if(change==='project')await h.click({openProject:'project-b'});
    if(change==='draft'){h.doc().shots[0].simplePrompt='changed';h.service.draft();}
    if(change==='new-request'){const next=h.click({service:'d3-preflight'});await flush();waiting[1](response(data));await next;}
    waiting[0](response(data));await checking;
    assert.equal(h.dialogs.length,['none','new-request'].includes(change)?1:0);
    if(h.dialogs.length)assert.match(h.dialogs.at(-1)[1],/镜头 shot-1 已按服务端/);
});

for(const outcome of ['success','error','empty','foreign'])test(`task completion preserves unrelated modal and remains accessible: ${outcome}`,async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});
    sessionStorage.setItem('t8director.activeJob:test',JSON.stringify({prompt_id:'original',project_id:outcome==='foreign'?'project-b':'project-a',shot_id:'shot-1'}));
    await h.service.restore();const watching=h.timers.shift()();await flush();
    const dialog=h.$('[data-dialog]');delete dialog.dataset.jobId;dialog.open=true;dialog.textContent='unsaved modal text';
    const count=h.dialogs.length,prior=fetch;
    globalThis.fetch=(url,opts)=>new URL(url).pathname.endsWith('/jobs/original')?Promise.resolve(response({state:outcome==='error'?'error':'success',outputs:['success','foreign'].includes(outcome)?{save:{videos:[{filename:'done.mp4'}]}}:{}})):prior(url,opts);
    await [...h.intervals.values()][0]();await watching;
    assert.equal(dialog.open,true);assert.equal(dialog.textContent,'unsaved modal text');assert.equal(h.dialogs.length,count);
    assert.equal(h.$('[data-service="job-status"]').hidden,false);
    dialog.close();await h.click({service:'job-status'});
    assert.equal(h.dialogs.length,count+1);assert.match(h.dialogs.at(-1)[1],/original/);
});

test('task status can close its own dialog without closing unrelated editors',async()=>{
    const h=serviceHarness({local:project(),specific:'project-a'});
    sessionStorage.setItem('t8director.activeJob:test',JSON.stringify({prompt_id:'original',project_id:'project-a',shot_id:'shot-1'}));
    await h.service.restore();const watching=h.timers.shift()();await flush();
    const dialog=h.$('[data-dialog]');assert.equal(dialog.dataset.jobId,'original');
    const prior=fetch;globalThis.fetch=(url,opts)=>new URL(url).pathname.endsWith('/jobs/original')?Promise.resolve(response({state:'success',outputs:{save:{videos:[{filename:'done.mp4'}]}}})):prior(url,opts);
    await [...h.intervals.values()][0]();await watching;assert.equal(dialog.open,false);
    const html=readFileSync(new URL('../web/director/index.html',import.meta.url),'utf8');
    const fn=html.slice(html.indexOf('function showDialog('),html.indexOf('\n',html.indexOf('function showDialog(')));
    assert.match(fn,/delete dialog.dataset.jobId/);
});

for(const selection of [null,'removed','pending'])test(`stage, selected card and zoom agree after clearing references: ${selection}`,()=>{
    const html=readFileSync(new URL('../web/director/index.html',import.meta.url),'utf8');
    const line=name=>html.slice(html.indexOf('function '+name+'('),html.indexOf('\n',html.indexOf('function '+name+'(')));
    const stage=html.slice(html.indexOf('function renderStage('),html.indexOf('function renderThumbs('));
    const zoom=html.slice(html.indexOf("case 'zoom':"),html.indexOf("case 'pair':"));
    const s={id:'shot',mode:'first',first:null,last:null,refs:[],tray:['pending'],selected:selection};
    const nodes=new Map(),opened=[],notices=[];
    const state={doc:{sharedRefs:[],shots:[s]},assets:new Map(['pending','removed'].map(id=>[id,{id,kind:'image',name:id+'.png',url:id+'.png',width:100,height:200}])),sh:()=>s,s,view:'input',pair:false,esc:String,tokenMap:()=>new Map([['pending','@image1']]),$:selector=>{if(!nodes.has(selector))nodes.set(selector,{});return nodes.get(selector)},notify:t=>notices.push(t),fullPreview:id=>opened.push(id)};
    vm.createContext(state);vm.runInContext(['refIds','bindings','trayItems','previewAssetId','imageTag','renderThumbs'].map(line).join('\n')+'\n'+stage,state);
    vm.runInContext("renderStage();renderThumbs();switch('zoom'){"+zoom+'}',state);
    assert.match(nodes.get('[data-stage]').innerHTML,/pending\.png/);assert.match(nodes.get('[data-thumbs]').innerHTML,/data-selected="true"/);
    assert.deepEqual(opened,['pending']);assert.equal(notices.length,0);
    s.tray=[];vm.runInContext("switch('zoom'){"+zoom+'}',state);assert.match(notices.at(-1),/先添加/);
});
