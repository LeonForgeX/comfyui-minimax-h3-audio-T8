const copy = value => structuredClone(value);
const clean = value => String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const newRow = () => ({ id: crypto.randomUUID(), name: "", strength: 1, enabled: false, source: "user" });
const freshTwoPass = resolution => ({ mode: "two_pass", preset: "standard_4plus4_v1", output_mp: resolution || "auto", upscaler: "auto", low_loras: [], high_loras: [] });

export function createSamplingDialog({ root, doc, shot, catalog, apply, notify }) {
    const dialog = root.querySelector("[data-model-zone]");
    const body = dialog.querySelector("[data-model-settings]");
    let draft = null;
    let scope = "global";
    let mode = "single";
    const filters = { loras: "", low_loras: "", high_loras: "" };
    const options = (kind, value, auto = false, filter = "") => {
        const available = [...(auto ? [{ value: "auto", label: "自动" }] : []), ...(catalog()[kind] || []).filter(item => item.value !== "auto" && item.value !== "none")];
        const list = available.filter(item => !filter || item.value.toLowerCase().includes(filter.toLowerCase()));
        if (value && !list.some(item => item.value === value)) list.push(available.find(item => item.value === value) || { value, label: `${value}（本机未找到）` });
        return list.map(item => `<option value="${clean(item.value)}" ${item.value === value ? "selected" : ""}>${clean(item.label)}</option>`).join("");
    };
    const selected = () => scope === "global" ? draft.global : draft.local;
    const defaultSingle = () => ({ mode: "single", resolution_mp: draft.generation.resolution_mp ?? "auto", lora_mode: draft.generation.lora_mode ?? "auto", loras: copy(draft.generation.loras || []) });
    const ensure = name => {
        const holder = selected();
        holder[name] ??= name === "single" ? defaultSingle() : freshTwoPass(draft.generation.resolution_mp);
        if (name === "single") {
            const fallback = defaultSingle();
            holder.single.resolution_mp ??= fallback.resolution_mp;
            holder.single.lora_mode ??= fallback.lora_mode;
            holder.single.loras ??= fallback.loras;
        }
        return holder[name];
    };
    function open() {
        const project = doc();
        const current = shot();
        const globalSampling = project.sampling || { mode: "single" };
        const localSampling = current.sampling || globalSampling;
        draft = {
            generation: copy(project.generation || {}),
            global: { mode: globalSampling.mode || "single", single: globalSampling.single && copy(globalSampling.single), two_pass: globalSampling.two_pass && copy(globalSampling.two_pass), [globalSampling.mode || "single"]: copy(globalSampling) },
            local: { mode: localSampling.mode || "single", single: localSampling.single && copy(localSampling.single), two_pass: localSampling.two_pass && copy(localSampling.two_pass), [localSampling.mode || "single"]: copy(localSampling) },
            inherited: current.samplingInherit !== false,
            localInitialized: current.samplingInherit === false,
        };
        scope = draft.inherited ? "global" : "local";
        mode = selected().mode;
        render();
        dialog.showModal();
    }
    function close() { draft = null; dialog.close(); }
    function render() {
        if (!draft) return;
        const target = ensure(mode);
        const shared = draft.generation;
        const single = mode === "single";
        const observedPatchRisk = !single && shared.unet === "minimax_h3_fl2va_pruned_int8_convrot.safetensors" &&
            [...(target.low_loras || []), ...(target.high_loras || [])].some(row => row.enabled && row.name === "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors");
        const loraSection = key => {
            const rows = target[key] || [];
            return `<section class="o-sampling-stage"><div class="o-row o-between"><h3>${key === "low_loras" ? "一采 · 基础画面" : key === "high_loras" ? "二采 · 高清细化" : "单采 LoRA"}</h3><button type="button" data-sampling-action="add" data-stage="${key}">＋ 添加 LoRA</button></div>
                <p class="o-tip">已启用 ${rows.filter(row => row.enabled).length} / 共 ${rows.length}；按列表顺序加载，重复文件不会合并。</p>
                <label><span>搜索 LoRA 文件／目录</span><input data-sampling-search="${key}" value="${clean(filters[key])}" placeholder="输入名称或目录的一部分"></label>
                ${rows.map((row, index) => `<div class="o-sampling-row" data-stage="${key}" data-index="${index}">
                    <label class="o-check"><input type="checkbox" data-sampling-field="enabled" ${row.enabled ? "checked" : ""}><span>启用</span></label>
                    <label><span>LoRA 文件</span><select data-sampling-field="name"><option value="">选择文件</option>${options("lora", row.name, false, filters[key])}</select></label>
                    <label><span>强度</span><input type="number" min="-2" max="2" step="0.05" data-sampling-field="strength" value="${clean(row.strength)}"></label>
                    <div class="o-row"><button type="button" data-sampling-action="up" aria-label="上移第 ${index + 1} 条">↑</button><button type="button" data-sampling-action="down" aria-label="下移第 ${index + 1} 条">↓</button><button type="button" data-sampling-action="remove" aria-label="移除第 ${index + 1} 条">×</button></div>
                </div>`).join("")}</section>`;
        };
        const following = doc().shots.filter(item => item.samplingInherit !== false).length;
        body.innerHTML = `<div class="o-row o-between"><div class="o-row"><button type="button" data-sampling-scope="global" aria-pressed="${scope === "global"}">全片默认 · ${following} 镜跟随</button><button type="button" data-sampling-scope="local" aria-pressed="${scope === "local"}">当前镜头独立</button></div><small>${scope === "global" ? "修改跟随全片的镜头；独立镜头不变" : "只修改当前镜头"}</small></div>
            ${scope === "global" && !draft.inherited ? '<p class="o-tip">正在定位全片配置错误；应用后当前镜头仍使用独立设置。若要改为跟随全片，请明确点击“全片默认”。</p>' : ""}
            <div class="o-sampling-shared o-gap"><h3>共用底模与编码器</h3><div class="o-two">${[["unet", "扩散模型"], ["clip", "文本编码器"], ["video_vae", "视频 VAE"], ["audio_vae", "音频 VAE"]].map(([key, label]) => `<label><span>${label}</span><select data-sampling-model="${key}">${options(key, shared[key] || "auto", true)}</select></label>`).join("")}</div><p class="o-tip">同一镜的一采与二采共用以上底模、CLIP 和 VAE；只加载一份基础 MODEL。</p></div>
            <div class="o-row o-gap"><span>采样方式</span><button type="button" data-sampling-mode="single" aria-pressed="${single}">单采样</button><button type="button" data-sampling-mode="two_pass" aria-pressed="${!single}">双采样 · 高清细化</button></div>
            <div class="o-two o-gap"><label><span>最终输出 / MP</span><select data-sampling-mp><option value="auto" ${String(single ? target.resolution_mp : target.output_mp) === "auto" ? "selected" : ""}>自动</option>${[0.4, 0.5, 0.6, 0.8, 1].map(value => `<option value="${value}" ${String(single ? target.resolution_mp : target.output_mp) === String(value) ? "selected" : ""}>${value} MP</option>`).join("")}</select></label>${single ? `<label><span>单采 LoRA 模式</span><select data-sampling-lora-mode>${[["auto", "自动 Turbo（旧行为）"], ["none", "不加载"], ["manual", "手动列表"]].map(([key, label]) => `<option value="${key}" ${target.lora_mode === key ? "selected" : ""}>${label}</option>`).join("")}</select></label>` : `<label><span>学习型 3D 放大模型</span><select data-sampling-upscaler>${options("upscaler", target.upscaler || "auto", true)}</select></label>`}</div>
            ${observedPatchRisk ? `<p class="o-warn">本机 GPU 实测该裁剪版底模＋EMA B LoRA 出现 AdaLN patch 形状错误。视频可能仍生成，但不能认定 LoRA 完整生效；保留您的选择，请查看 Core 日志或改用完整底模复测。</p>` : ""}
            ${single ? loraSection("loras") : `<p class="o-tip">标准 4+4：LOW 预测 → 学习型 3D 放大 → HIGH 细化。两路 LoRA 均从未打补丁的同一底模独立派生；不暗中加入 Turbo。</p><div class="o-sampling-columns">${loraSection("low_loras")}${loraSection("high_loras")}</div><button type="button" data-sampling-action="copy-low">复制一采列表到二采（替换）</button>`}`;
    }
    dialog.addEventListener("click", event => {
        const button = event.target.closest("button");
        if (!button || !draft) return;
        if (button.dataset.samplingScope) {
            scope = button.dataset.samplingScope;
            draft.inherited = scope === "global";
            if (scope === "local" && !draft.localInitialized) {
                draft.local = copy(draft.global);
                draft.localInitialized = true;
            }
            mode = selected().mode;
            render();
            return;
        }
        if (button.dataset.samplingMode) {
            mode = button.dataset.samplingMode;
            selected().mode = mode;
            render();
            return;
        }
        const action = button.dataset.samplingAction;
        if (!action) return;
        const target = ensure(mode);
        if (action === "copy-low") {
            if (target.high_loras.length && !confirm("用一采列表替换现有二采 LoRA？")) return;
            target.high_loras = target.low_loras.map(row => ({ ...copy(row), id: crypto.randomUUID() }));
        } else {
            const key = button.dataset.stage || button.closest("[data-stage]")?.dataset.stage;
            const rows = target[key];
            const index = Number(button.closest("[data-index]")?.dataset.index);
            if (!rows) return;
            if (action === "add") rows.push(newRow());
            else if (action === "remove") rows.splice(index, 1);
            else if (action === "up" && index > 0) [rows[index - 1], rows[index]] = [rows[index], rows[index - 1]];
            else if (action === "down" && index < rows.length - 1) [rows[index + 1], rows[index]] = [rows[index], rows[index + 1]];
        }
        render();
    });
    dialog.addEventListener("change", event => {
        if (!draft) return;
        const el = event.target;
        if (el.dataset.samplingModel) { draft.generation[el.dataset.samplingModel] = el.value; return; }
        const target = ensure(mode);
        if (el.hasAttribute("data-sampling-mp")) { target[mode === "single" ? "resolution_mp" : "output_mp"] = el.value; return; }
        if (el.hasAttribute("data-sampling-upscaler")) { target.upscaler = el.value; return; }
        if (el.hasAttribute("data-sampling-lora-mode")) { target.lora_mode = el.value; return; }
        const key = el.closest("[data-stage]")?.dataset.stage;
        const index = Number(el.closest("[data-index]")?.dataset.index);
        if (!key || !Number.isInteger(index) || !target[key]?.[index]) return;
        const row = target[key][index];
        const field = el.dataset.samplingField;
        if (field === "enabled") row.enabled = el.checked;
        else if (field === "name") row.name = el.value;
        else if (field === "strength") row.strength = el.value.trim() === "" ? null : Number(el.value);
        render();
    });
    dialog.addEventListener("input", event => {
        const key = event.target.dataset.samplingSearch;
        if (!key || !draft) return;
        const cursor = event.target.selectionStart;
        filters[key] = event.target.value;
        render();
        const field = dialog.querySelector(`[data-sampling-search="${key}"]`);
        field?.focus();
        field?.setSelectionRange(cursor, cursor);
    });
    dialog.addEventListener("cancel", () => { draft = null; });
    function commit() {
        if (!draft) return;
        // Error navigation changes the displayed scope, not the user's inheritance choice.
        // Global is always applied; local is applied only when explicitly selected.
        // Inactive single/two-pass drafts are retained, not executed or validated.
        for (const owner of draft.inherited ? ["global"] : ["global", "local"]) {
            const holder = draft[owner];
            const stages = holder.mode === "single" ? ["loras"] : ["low_loras", "high_loras"];
            for (const key of stages) {
                const rows = holder[holder.mode]?.[key] || [];
                const invalidStrength = row => !Number.isFinite(row.strength) || row.strength < -2 || row.strength > 2;
                const index = rows.findIndex(row => invalidStrength(row) || (row.enabled && !row.name));
                if (index < 0) continue;
                scope = owner; mode = holder.mode; render();
                const strengthError = invalidStrength(rows[index]);
                const message = `${owner === "global" ? "全片默认" : "当前镜头"} · ${{loras:"单采",low_loras:"一采",high_loras:"二采"}[key]}第 ${index + 1} 条：${strengthError ? "LoRA 强度必须是 -2–2 的有限数字（禁用行也需有效）" : "已启用的 LoRA 必须选择文件"}`;
                const rowSelector = `[data-stage="${key}"][data-index="${index}"]`;
                const field = dialog.querySelector(`${rowSelector} [data-sampling-field="${strengthError ? "strength" : "name"}"]`);
                const errorId = `sampling-error-${key}-${index}`;
                (dialog.querySelector(rowSelector) || body).insertAdjacentHTML("afterbegin", `<p id="${errorId}" role="alert" class="o-warn" style="grid-column:1/-1">${clean(message)}</p>`);
                field?.setAttribute("aria-invalid", "true");
                field?.setAttribute("aria-describedby", errorId);
                field?.focus();
                return;
            }
        }
        const shared = draft.generation;
        const saved = holder => {
            const active = copy(holder[holder.mode]);
            delete active.single;
            delete active.two_pass;
            const single = holder.single && copy(holder.single);
            const two_pass = holder.two_pass && copy(holder.two_pass);
            if (single) { delete single.single; delete single.two_pass; active.single = single; }
            if (two_pass) { delete two_pass.single; delete two_pass.two_pass; active.two_pass = two_pass; }
            return active;
        };
        const inherited = draft.inherited;
        apply({ generation: shared, global: saved(draft.global), local: saved(draft.local), inherited });
        close();
        notify(inherited ? "全片采样设置已应用；独立镜头未改动。" : "当前镜头独立采样设置已应用。");
    }
    return { open, close, render, commit };
}
