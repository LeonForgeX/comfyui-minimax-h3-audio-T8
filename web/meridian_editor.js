import { app } from "../../scripts/app.js";

// Local canvas authoring only. No CDN, GPU requests, timers or global queue operations.
const ID = "MiniMaxH3MeridianCameraEXPT8";
const PRESETS = ["slide","push_in","pull_out","crane","orbit","freeze_orbit","source_camera"];
const clone = (v) => JSON.parse(JSON.stringify(v));
export const stable = (v) => JSON.stringify(v, (_, value) => value && !Array.isArray(value) && typeof value === "object"
    ? Object.fromEntries(Object.keys(value).sort().map((key) => [key,value[key]])) : value);
const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (a) => { const n = Math.hypot(...a); return n > 1e-8 ? a.map((v) => v/n) : null; };

export function validateDraft(plan, material) {
    const exact = (v, keys) => v && typeof v === "object" && !Array.isArray(v) &&
        Object.keys(v).sort().join(",") === keys.sort().join(",");
    if (!exact(plan,["schema","geometry_id","origin_frame","window_end","frames","space","units","roll","mode","pivot","camera_keys","time_keys"]) ||
        plan.space!=="source_window_first_camera_x_right_y_down_z_forward" || plan.units!=="fixed_pivot_depth" ||
        !Number.isInteger(plan.origin_frame) || !Number.isInteger(plan.window_end) ||
        !Array.isArray(plan.pivot) || plan.pivot.length!==3 || plan.pivot.some(v=>!Number.isFinite(v))) throw Error("规范字段、坐标系或深度单位不合法。");
    if (plan?.schema !== "t8.meridian.camera.v1" || plan.geometry_id !== material.identity ||
        plan.origin_frame !== material.start || plan.window_end !== material.end) throw Error("素材或窗口已变，请显式重置旧路径。");
    if (!Number.isInteger(plan.frames) || plan.frames < 2 || plan.roll !== 0 ||
        !["authored", "source_camera"].includes(plan.mode)) throw Error("输出帧数／模式／roll不合法。");
    for (const name of ["camera_keys", "time_keys"]) {
        const keys = plan[name];
        if (!Array.isArray(keys) || keys.length < 2 || keys.length > 4096 || keys[0].t !== 0 || keys.at(-1).t !== plan.frames-1 ||
            keys.some((k, i) => !Number.isInteger(k.t) || (i && k.t <= keys[i-1].t))) throw Error("两个轨的关键帧必须独立覆盖0..N-1，且严格递增。");
    }
    for (const k of plan.camera_keys) {
        if(!exact(k,["t","pos","look","focal","ease"]))throw Error("空间关键帧字段不完整或含未知项。");
        for (const p of [k.pos, k.look]) if (!Array.isArray(p) || p.length !== 3 || p.some((v) => !Number.isFinite(v))) throw Error("相机坐标必须是三个有限数。");
        if (Math.hypot(...sub(k.pos, k.look)) < 1e-6 || !Number.isFinite(k.focal) || k.focal <= 0 || typeof k.ease !== "boolean") throw Error("位置／目标／焦距／缓动不合法。");
    }
    for (let i = 0; i < plan.time_keys.length; i++) {
        if(!exact(plan.time_keys[i],["t","src"]))throw Error("时间关键帧字段不完整或含未知项。");
        const s = plan.time_keys[i].src;
        if (!Number.isInteger(s) || s < material.start || s > material.end || (i && s < plan.time_keys[i-1].src) ||
            (material.kind === "image" && s !== material.start)) throw Error("源时间越界或逆向；IMAGE必须冻结。");
    }
    return plan;
}

export function setDraft(node, plan, material) {
    validateDraft(plan, material);
    const target = widget(node, "camera_plan");
    if (!target) throw Error("找不到规范路径控件。");
    // The actual serialized API widget is the source of truth. No unsaved hidden draft.
    target.value = JSON.stringify(plan);
    const frames = widget(node, "frames");
    if (frames) frames.value = plan.frames;
    node.setDirtyCanvas?.(true, true);
    return target.value;
}

function element(tag, text, parent) {
    const el = document.createElement(tag);
    if (text != null) el.textContent = text;
    parent?.appendChild(el);
    return el;
}

export function setPresetDraft(node, draft) {
    if (!PRESETS.includes(draft.preset) || !Number.isFinite(draft.strength) ||
        !Number.isInteger(draft.frames) || draft.frames<22 || (draft.frames-5)%17) {
        throw Error("预设、有限强度或17k+5帧数不合法；长度还需对应native assets，无人为帧数上限。");
    }
    for (const name of ["preset","strength","frames","camera_plan"]) {
        if (!widget(node,name)) throw Error(`找不到${name}工作流控件。`);
    }
    for (const name of ["preset","strength","frames"]) widget(node,name).value=draft[name];
    // Applying is an explicit preset reset, not binding fictitious geometry to JSON.
    widget(node,"camera_plan").value="";
    node._t8MeridianPayload=null;
    node.setDirtyCanvas?.(true,true);
}

function openPresetEditor(node) {
    node._t8MeridianClose?.();
    const dialog=element("dialog",null,document.body);
    dialog.setAttribute("aria-label","Meridian二维预设规划（无几何）");
    dialog.style.cssText="box-sizing:border-box;width:min(900px,95vw);max-height:92vh;overflow:auto;padding:24px;background:#151d29;color:#eef3fb;border:1px solid #61738b;border-radius:12px;font:16px/1.6 system-ui";
    element("h2","Meridian · 二维预设规划",dialog);
    element("p","这里没有真实几何、深度或生成预览，只展示运镜方向示意。不会加载模型或运行GPU。应用后只保存节点preset／strength／frames，明确清空旧规范路径；真实三维空间和源时间轨需运行预览模板后编辑。",dialog).style.color="#f8cc80";
    const select=element("select",null,dialog);select.setAttribute("aria-label","运镜预设");
    for(const name of PRESETS){const option=element("option",name,select);option.value=name;}
    select.value=widget(node,"preset")?.value??"slide";
    const row=element("div",null,dialog);row.style.cssText="display:flex;gap:24px;flex-wrap:wrap;margin:20px 0";
    function field(name,text){const label=element("label",text+" ",row),input=element("input",null,label);input.type="number";input.setAttribute("aria-label",text);input.value=widget(node,name)?.value??(name==="frames"?73:.08);input.step=name==="frames"?1:.01;input.style.cssText="width:150px;font:inherit";return input;}
    const strength=field("strength","相对深度位移／角度强度"),frames=field("frames","输出帧数");
    const diagram=element("div",null,dialog);diagram.style.cssText="padding:32px;text-align:center;border:1px dashed #61738b;font:22px/1.8 system-ui;background:#243043";
    const descriptions={slide:"相机平行横移 → 朝向保持不变，不锁定目标",push_in:"相机向目标推进 ↑",pull_out:"相机远离目标 ↓",crane:"相机升高 ↑ 固定目标",orbit:"相机绕目标转动 ↻",freeze_orbit:"源时刻冻结 ○ 相机环绕 ↻",source_camera:"跟随实际估计源相机路径 →"};
    const draw=()=>{diagram.textContent="二维方向示意（非真实投影）： "+descriptions[select.value];};
    select.addEventListener("change",draw);draw();
    element("p","强度单位在真实几何后才按中央有效深度确定；orbit为strength×100度。源窗口由素材节点设置，原声1:1政策由后端严格检查。缺少对应长度assets时明确报错，不补帧冒充。",dialog);
    const status=element("p","",dialog),apply=element("button","应用预设（明确清空旧规范路径）",dialog),close=element("button","关闭，不应用",dialog);
    apply.addEventListener("click",()=>{try{setPresetDraft(node,{preset:select.value,strength:Number(strength.value),frames:Number(frames.value)});cleanup();}catch(err){status.textContent=err.message;}});
    const cleanup=()=>{dialog.remove();if(node._t8MeridianClose===cleanup)node._t8MeridianClose=null;};
    node._t8MeridianClose=cleanup;close.addEventListener("click",cleanup);dialog.addEventListener("cancel",cleanup);dialog.addEventListener("close",cleanup);dialog.showModal();
}

function openEditor(node) {
    node._t8MeridianClose?.();
    const material = node._t8MeridianPayload;
    if (!material?.plan) {
        openPresetEditor(node);
        return;
    }
    let plan;
    try { plan = validateDraft(JSON.parse(widget(node, "camera_plan")?.value || JSON.stringify(material.plan)), material); }
    catch (err) { alert(err.message); return; }
    const dialog = document.createElement("dialog");
    dialog.setAttribute("aria-label", "Meridian空间与时间运镜编辑器");
    dialog.style.cssText = "box-sizing:border-box;width:min(1220px,95vw);max-height:94vh;overflow:auto;padding:22px;background:#151d29;color:#eef3fb;border:1px solid #61738b;border-radius:12px;font:14px/1.5 system-ui";
    document.body.appendChild(dialog);
    const heading = element("div", null, dialog);
    heading.style.cssText = "display:flex;gap:18px;align-items:center;justify-content:space-between";
    element("h2", "Meridian · 独立空间轨／源时间轨", heading);
    const close = element("button", "关闭（所有有效修改已写入工作流）", heading);
    const notice = element("p", "几何预览不是生成视频。固定首源相机坐标：x右、y下、z前；固定深度单位，roll=0。目标视图只显示所选关键帧，真正Hermite轨迹与源取样由后端计算。", dialog);
    notice.style.color = "#f8cc80";
    const status = element("p", "", dialog);
    const grids = element("div", null, dialog);
    grids.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px";
    const source = element("section", null, grids);
    element("h3", "源素材（最多8张抽样缩略图）", source);
    const image = element("img", null, source);
    image.style.cssText = "width:100%;max-height:350px;object-fit:contain;background:#000";
    const sourceLabel = element("p", "", source);
    const sourceSlider = element("input", null, source);
    sourceSlider.type = "range"; sourceSlider.min = 0; sourceSlider.max = material.clouds.length-1; sourceSlider.value = 0;
    sourceSlider.style.width = "100%";
    const target = element("section", null, grids);
    element("h3", "抽样点云／目标关键帧视图（可点选目标）", target);
    const canvas = element("canvas", null, target);
    canvas.width = 620; canvas.height = 350;
    canvas.style.cssText = "width:100%;height:350px;background:#000;touch-action:none";
    const targetLabel = element("p", "", target);
    let selected = 0, pointsOnScreen = [];
    const cameraControls = element("section", null, dialog);
    element("h3", "空间轨 · 每行独立位置／目标／焦距／缓动", cameraControls);
    const cameraRows = element("div", null, cameraControls);
    const addCamera = element("button", "添加空间关键帧", cameraControls);
    const timeControls = element("section", null, dialog);
    element("h3", "源时间轨 · 输出帧t → 24fps源索引src", timeControls);
    element("p", `源窗口${material.start}..${material.end}。相同src可冻结，逐帧+1才可沿用原声；不允许逆向。这里的时间不会改变空间轨切线。`, timeControls);
    const timeRows = element("div", null, timeControls);
    const addTime = element("button", "添加时间关键帧", timeControls);
    const footer = element("section", null, dialog);
    footer.style.cssText = "display:flex;gap:12px;flex-wrap:wrap;margin-top:18px";
    const exportButton = element("button", "导出规范JSON", footer);
    const importInput = element("input", null, footer); importInput.type = "file"; importInput.accept = ".json,application/json";
    const reset = element("button", "恢复本次后端路径", footer);
    const preset = element("button", "清空路径，下一次使用节点预设", footer);
    let alive = true;
    const commit = () => {
        try {
            setDraft(node, plan, material);
            status.textContent = "已写入工作流/API路径。修改后请运行此运镜节点，后端会校验并重建真实warp；不会在后台启动GPU。";
            status.style.color = "#8ee4b5";
            render();
            return true;
        } catch (err) { status.textContent = err.message + " 无效修改未写入工作流。"; status.style.color = "#ff9c91"; return false; }
    };
    function render() {
        const cloud = material.clouds[Number(sourceSlider.value)];
        image.src = cloud.thumbnail;
        sourceLabel.textContent = `显示抽样源帧${cloud.src}；不是所有源帧。训练ROI输出${material.canvas.join("×")}，不拉伸。`;
        const key = plan.camera_keys[selected];
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#000"; ctx.fillRect(0,0,canvas.width,canvas.height);
        const f = norm(sub(key.look, key.pos));
        let r = f && norm(cross(f, [0,-1,0]));
        if (!r) r = [1,0,0];
        const d = f && cross(f,r);
        pointsOnScreen = [];
        if (!f) return;
        const localIndex = cloud.src-material.start;
        const K = material.source_intrinsics[localIndex];
        const focal = K[0][0]*key.focal/512*canvas.width;
        cloud.points.forEach((p,i) => {
            const v = sub(p,key.pos), z = dot(v,f);
            if (z <= .001) return;
            const x = canvas.width/2 + focal*dot(v,r)/z, y = canvas.height/2 + focal*dot(v,d)/z;
            if (x < 0 || x > canvas.width || y < 0 || y > canvas.height) return;
            const rgb = cloud.colors[i];
            ctx.fillStyle = `rgb(${rgb.join(",")})`; ctx.fillRect(x-2,y-2,4,4);
            pointsOnScreen.push({ x,y,p });
        });
        const fov = 2*Math.atan(512/(2*K[0][0]*key.focal))*180/Math.PI;
        targetLabel.textContent = `所选空间关键帧t=${key.t}；512方形几何水平FOV≈${fov.toFixed(1)}°（非最终ROI FOV），点击真实抽样点改变该关键帧look。`;
    }
    function input(row, value, name, onChange, integer = false) {
        const label = element("label", name + " ", row);
        const field = element("input", null, label);
        field.type = "number"; field.value = value; field.step = integer ? "1" : ".01";
        field.style.cssText = "width:75px;box-sizing:border-box;margin-right:6px";
        field.addEventListener("change", () => {
            const v = Number(field.value), previous = clone(plan);
            if (!Number.isFinite(v) || (integer && !Number.isInteger(v))) { field.value = value; return; }
            onChange(v);
            if (!commit()) { plan = previous; rows(); }
        });
    }
    function rows() {
        cameraRows.replaceChildren(); timeRows.replaceChildren();
        plan.camera_keys.forEach((k,i) => {
            const row = element("div", null, cameraRows);
            row.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:10px 0;padding:10px;background:#243043";
            const choose = element("button", i===selected ? "正在预览" : "预览", row);
            choose.addEventListener("click", () => { selected=i; rows(); render(); });
            input(row,k.t,"t",v=>k.t=v,true);
            for (const p of ["pos","look"]) for (let axis=0;axis<3;axis++) input(row,k[p][axis],`${p}.${"xyz"[axis]}`,v=>k[p][axis]=v);
            input(row,k.focal,"焦距倍率",v=>k.focal=v);
            const ease = element("label", "离开此帧缓动 ", row), box = element("input",null,ease);
            box.type="checkbox";box.checked=k.ease;
            box.addEventListener("change",()=> { k.ease=box.checked; commit(); });
            if (i>0 && i<plan.camera_keys.length-1) {
                const remove = element("button","删除",row);
                remove.addEventListener("click",()=> { plan.camera_keys.splice(i,1);selected=Math.min(selected,plan.camera_keys.length-1);commit();rows(); });
            }
        });
        plan.time_keys.forEach((k,i) => {
            const row = element("div",null,timeRows);row.style.cssText="display:flex;gap:12px;margin:10px 0";
            input(row,k.t,"输出t",v=>k.t=v,true);input(row,k.src,"源src",v=>k.src=v,true);
            if (i>0 && i<plan.time_keys.length-1) {
                const remove=element("button","删除",row);remove.addEventListener("click",()=>{plan.time_keys.splice(i,1);commit();rows();});
            }
        });
    }
    function addKey(name) {
        const keys=plan[name];
        let interval=0;
        for(let i=0;i<keys.length-1;i++) if(keys[i+1].t-keys[i].t>keys[interval+1].t-keys[interval].t)interval=i;
        if(keys[interval+1].t-keys[interval].t<2) { status.textContent="输出帧已经没有可插入整数位置。";return; }
        const first=keys[interval],last=keys[interval+1],key=clone(first);
        key.t=Math.floor((first.t+last.t)/2);
        if(name==="time_keys")key.src=Math.floor((first.src+last.src)/2);
        // Space insertion is an explicit authoring edit, not a claim to preserve the prior Hermite curve.
        keys.splice(interval+1,0,key);if(commit())rows();
    }
    addCamera.addEventListener("click",()=>addKey("camera_keys"));
    addTime.addEventListener("click",()=>addKey("time_keys"));
    sourceSlider.addEventListener("input",render);
    canvas.addEventListener("pointerdown",event=>{
        const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height;
        let best=null, distance=20;
        for(const point of pointsOnScreen){const delta=Math.hypot(x-point.x,y-point.y);if(delta<distance){best=point;distance=delta;}}
        if(best){const previous=clone(plan);plan.camera_keys[selected].look=clone(best.p);if(!commit())plan=previous;rows();}
    });
    exportButton.addEventListener("click",()=>{
        if(!commit())return;
        const url=URL.createObjectURL(new Blob([JSON.stringify(plan,null,2)],{type:"application/json"}));
        const link=element("a",null);link.href=url;link.download="Meridian_camera_plan.json";link.click();URL.revokeObjectURL(url);
    });
    importInput.addEventListener("change",async()=>{
        const file=importInput.files?.[0];if(!file)return;
        if(file.size>1024*1024){status.textContent="导入文件超过1MiB。";return;}
        try {const imported=validateDraft(JSON.parse(await file.text()),material);if(!alive)return;plan=clone(imported);selected=0;commit();rows();}
        catch(err){if(alive)status.textContent=err.message;}
    });
    reset.addEventListener("click",()=>{plan=clone(material.plan);selected=0;commit();rows();});
    preset.addEventListener("click",()=>{widget(node,"camera_plan").value="";node.setDirtyCanvas?.(true,true);node._t8MeridianClose?.();});
    const cleanup=()=>{alive=false;dialog.remove();if(node._t8MeridianClose===cleanup)node._t8MeridianClose=null;};
    node._t8MeridianClose=cleanup;close.addEventListener("click",cleanup);
    dialog.addEventListener("cancel",cleanup);dialog.addEventListener("close",cleanup);
    rows();render();dialog.showModal();
}

app.registerExtension({
    name:"minimax-h3-audio-t8.meridian-editor",
    async beforeRegisterNodeDef(nodeType,nodeData) {
        if(nodeData.name!==ID)return;
        const created=nodeType.prototype.onNodeCreated,executed=nodeType.prototype.onExecuted,removed=nodeType.prototype.onRemoved;
        nodeType.prototype.onNodeCreated=function(){
            const result=created?.apply(this,arguments);
            this.addWidget("button","打开大号运镜／时间编辑器",null,()=>openEditor(this),{serialize:false});
            this.addWidget("button","重置规范路径，使用节点预设",null,()=>{
                this._t8MeridianClose?.();const w=widget(this,"camera_plan");if(w)w.value="";this.setDirtyCanvas?.(true,true);
            },{serialize:false});
            this.addWidget("button","二维预设规划（不运行GPU）",null,()=>openPresetEditor(this),{serialize:false});
            this.setSize?.([Math.max(this.size?.[0]??0,620),Math.max(this.size?.[1]??0,450)]);
            return result;
        };
        nodeType.prototype.onExecuted=function(message){
            const result=executed?.apply(this,arguments),payload=message?.meridian_editor?.[0];
            if(!payload?.plan)return result;
            const current=widget(this,"camera_plan")?.value;
            // A changed draft cannot be overwritten by an older completed GPU/CPU preview.
            if(current){try{if(stable(JSON.parse(current))!==stable(payload.plan))return result;}catch{return result;}}
            this._t8MeridianPayload=clone(payload);
            return result;
        };
        nodeType.prototype.onRemoved=function(){this._t8MeridianClose?.();this._t8MeridianPayload=null;return removed?.apply(this,arguments);};
    }
});
