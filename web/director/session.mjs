// Persistent D1 services; UI drafts are separate from saved server revisions.
export function directorUUID() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    // getRandomValues is available on ordinary LAN HTTP, unlike randomUUID.
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    const hex = [...bytes].map(b => b.toString(16).padStart(2, "0"));
    return [hex.slice(0,4), hex.slice(4,6), hex.slice(6,8), hex.slice(8,10), hex.slice(10)].map(a=>a.join("")).join("-");
}
export function directorOutputVideos(data) {
    const result = [];
    for (const value of Object.values(data?.outputs || {})) {
        for (const key of ["videos", "gifs", "video"]) {
            if (Array.isArray(value?.[key])) result.push(...value[key]);
        }
        // SafeAVSave reports MP4 under Core's legacy `images` key.
        if (Array.isArray(value?.images)) result.push(...value.images.filter(item => /\.(mp4|mov|mkv|webm)$/i.test(item?.filename || "")));
    }
    return result;
}
export function makeDirectorServices(ctx) {
    const { $, esc, notify, showDialog } = ctx;
    // The HTML prototype is intentionally kept as the stable D1 layout.  At
    // runtime the real service owns the badge/copy so users cannot mistake the
    // current D2a–D2c queue bridge for the old preflight-only page.
    const badge = $(".o-badge");
    if (badge) badge.textContent = "D2a–D2c · 真实生成";
    const note = $(".o-note");
    if (note) note.textContent = "D2a–D2c：文字、首帧、首尾、参考素材、原音驱动和参考音色均按正式 Core 配方编译；GPU画质仍需逐项验收。";
    const generationButton = $("[data-action=\"generate\"]");
    if (generationButton) {
        const normalizeLabel = () => {
            if (/编译预检|生成第/.test(generationButton.textContent || "")) {
                generationButton.textContent = "生成当前镜头";
            }
        };
        new MutationObserver(normalizeLabel).observe(generationButton, { childList: true, characterData: true, subtree: true });
        normalizeLabel();
    }
    const base = new URL(".", import.meta.url);
    const id = directorUUID;
    const onlineNote = "D2a–D2c：文字、首帧、首尾、参考素材、原音驱动和参考音色均按正式 Core 配方编译；GPU画质仍需逐项验收。";
    let tab = sessionStorage.getItem("t8director.tab");
    if (!tab) { tab = id(); sessionStorage.setItem("t8director.tab", tab); }
    const draftKey = "t8director.draft:" + tab;
    let projectId = id(), revision = 0, title = "我的第一部短片", busy = false, reconnectId = null;
    let projectEpoch = 0;
    let resultRequest = 0, compileRequest = 0, d3Request = 0, projectLoadRequest = 0, editEpoch = 0;
    let reconnectTarget = null, lastJobDialog = null;
    const completedResults = new Map();
    const contextToken = () => `${projectId}:${projectEpoch}`;
    let latest = null, importFile = null, activeJobId = null, activeUpload = null, stopWatchingJob = null, cancelledJobId = null;
    const activeJobKey = "t8director.activeJob:" + tab;
    const assetURL = aid => new URL("assets/" + aid, base).href;
    const connectionNotice = () => {
        const online = navigator.onLine !== false;
        const badge = $(".o-badge");
        if (badge && !activeJobId) badge.textContent = online ? "D2a–D2c · 真实生成" : "离线 · 草稿仍保留";
        const note = $(".o-note");
        if (note) note.textContent = online ? onlineNote : "当前浏览器离线：编辑仍保存在本标签草稿，保存／上传／生成会在连接恢复后重试；已有任务不会重复提交。";
    };
    window.addEventListener("online", connectionNotice);
    window.addEventListener("offline", connectionNotice);
    connectionNotice();
    function uploadNotice(message, cancellable = false) {
        notify(message);
        const notice = $("[data-notice]");
        if (!notice) return;
        notice.querySelector("[data-upload-cancel]")?.remove();
        if (!cancellable) return;
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.service = "cancel-upload";
        button.dataset.uploadCancel = "";
        button.textContent = "取消上传";
        button.title = "取消网络上传；未确认的素材不会加入本镜";
        notice.append(" ", button);
    }
    function rememberJob(promptId, recipe, shotId, owner) {
        try { sessionStorage.setItem(activeJobKey, JSON.stringify({ prompt_id: promptId, recipe: recipe || "director", project_id: owner, shot_id: shotId, started_at: Date.now() })); }
        catch { /* Storage can be disabled; in-memory polling still remains safe. */ }
    }
    function pendingJob() {
        try {
            const value = JSON.parse(sessionStorage.getItem(activeJobKey) || "null");
            return value && typeof value.prompt_id === "string" ? value : null;
        } catch { return null; }
    }
    function showJobDialog(promptId, heading, body, force = false) {
        const dialog = $("[data-dialog]");
        // Background task updates must not replace an editor, library or confirmation.
        if (!force && dialog.open && !dialog.dataset.jobId) return;
        showDialog(heading, body);
        dialog.dataset.jobId = promptId;
    }
    function showActiveJob(force = true) {
        if (!activeJobId) {
            if (lastJobDialog) showJobDialog(...lastJobDialog, force);
            return;
        }
        const owner = pendingJob()?.project_id;
        showJobDialog(activeJobId, "导演台任务进行中", `<p>任务 ID：${esc(activeJobId)}</p><p>所属项目：${esc(owner || "旧记录未标注")}${owner && owner !== projectId ? "（不是当前项目）" : ""}</p><p data-job-state>正在跟踪原任务，不会重复提交。</p><button data-service="cancel-job">取消此任务</button>`, force);
    }
    function forgetJob(promptId = activeJobId) {
        try {
            const value = pendingJob();
            if (!promptId || !value || value.prompt_id === promptId) sessionStorage.removeItem(activeJobKey);
        } catch { /* Best effort only; the Core prompt remains the source of truth. */ }
    }
    const materialize = a => ({ ...a, url: assetURL(a.id), file: { arrayBuffer: async () => {
        const response = await fetch(assetURL(a.id));
        if (!response.ok) throw Error("服务端素材缺失，请重新上传并重连");
        return response.arrayBuffer();
    } } });
    const envelope = (selectedCurrent = ctx.current()) => {
        const aliasMaps = {};
        for (const s of ctx.doc().shots) aliasMaps[s.id] = Object.fromEntries([...ctx.tokenMap(s)].map(([aid, alias]) => [alias, aid]));
        return { schema: "t8.minimax_h3.director_project", version: 2, id: projectId, revision, title,
            doc: structuredClone(ctx.doc()), current: selectedCurrent, aliasMaps,
            assets: [...ctx.assets().values()].map(a => Object.fromEntries(Object.entries(a).filter(([key]) => !["url", "file", "peaks", "missing"].includes(key)))) };
    };
    async function request(path, body, method = body ? "POST" : "GET") {
        const response = await fetch(new URL(path, base), { method, credentials: "same-origin",
            ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}) });
        const data = await response.json();
        if (!response.ok) { const e = Error(data.error || "服务端请求失败"); e.status = response.status; e.data = data; throw e; }
        return data;
    }
    async function loadResults(owner = projectId, autoOpen = false) {
        const epoch = projectEpoch, sequence = ++resultRequest;
        const shots = ctx.doc().shots.map(shot => ["shot", shot.id]);
        const query = new URLSearchParams(shots).toString();
        const data = await request("results/" + encodeURIComponent(owner) + (query ? "?" + query : ""));
        if (owner === projectId && epoch === projectEpoch && sequence === resultRequest) {
            // A lagging server index cannot erase a completion already observed here.
            for (const item of data.results || []) {
                if (completedResults.get(item.shot_id)?.prompt_id === item.prompt_id && directorOutputVideos(item).length) completedResults.delete(item.shot_id);
            }
            const observed = [...completedResults.values()];
            ctx.setResults([...observed, ...(data.results || []).filter(item => !completedResults.has(item.shot_id))], autoOpen);
        }
        return data.results || [];
    }
    function draft() {
        latest = null;
        try { localStorage.setItem(draftKey, JSON.stringify(envelope())); $("[data-save]").textContent = `本标签草稿已恢复保护 · 服务端版本 ${revision} · 请保存项目`; }
        catch { $("[data-save]").textContent = "浏览器草稿保存失败：可先导出 project.json，再重试服务端保存"; }
    }
    function syncProjectURL() {
        const url = new URL(location.href);
        url.searchParams.set("project_id", projectId);
        window.history.replaceState(window.history.state, "", url.href);
    }
    function beginProjectLoad() {
        return { sequence: ++projectLoadRequest, context: contextToken(), snapshot: JSON.stringify(envelope()), edits: editEpoch };
    }
    function acceptProjectLoad(ticket) {
        if (ticket.sequence !== projectLoadRequest || ticket.context !== contextToken()) return false;
        if (ticket.edits !== editEpoch || ticket.snapshot !== JSON.stringify(envelope())) {
            notify("载入期间有新编辑，已保留当前草稿；请保存后重新打开或导入项目。");
            return false;
        }
        return true;
    }
    function retainCurrentDraft() {
        localStorage.setItem(draftKey + ":backup:" + projectId, JSON.stringify(envelope()));
    }
    function hydrate(p) {
        if (p.schema !== "t8.minimax_h3.director_project" || ![0, 1, 2].includes(p.version) || !Array.isArray(p.doc?.shots) || !p.doc.shots.length) throw Error("不是支持的导演台项目，保留原文件");
        // Never use client-supplied URLs, filesystem paths or executable code.
        projectEpoch++;
        completedResults.clear();
        projectId = p.id; revision = p.revision; title = p.title;
        const restored = structuredClone(p.doc);
        restored.sampling ??= { mode: "single" };
        ctx.replace(restored, p.current, new Map(p.assets.map(a => [a.id, materialize(a)])));
        latest = null; ctx.resetHistory(); ctx.render();
        loadResults(projectId, true).catch(error => notify("镜头成片记录暂时无法读取：" + error.message));
        $("[data-project-title]").value = title;
        $("[data-save]").textContent = `已载入项目 · 服务端版本 ${revision}`;
        localStorage.setItem("t8director.lastProject", projectId);
        syncProjectURL();
    }
    function download(name, value) {
        downloadBytes(name, JSON.stringify(value, null, 2));
    }
    function downloadBytes(name, bytes) {
        const url = URL.createObjectURL(new Blob([bytes], { type: "application/json" }));
        const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    async function save(copy = false) {
        if (busy) return;
        busy = true; $("[data-save]").textContent = "正在保存到服务端…";
        const epoch = projectEpoch;
        try {
            const before = envelope();
            const p = copy ? { ...before, id: id(), revision: 0, title: before.title + " · 副本" } : before;
            const saved = await request("projects/" + p.id, { project: p, expected_revision: p.revision });
            const savedProject = { ...p, revision: saved.revision };
            // A response belongs to the document that initiated it, never a subsequently opened project.
            if (epoch !== projectEpoch) {
                localStorage.setItem(draftKey + ":saved:" + p.id, JSON.stringify(savedProject));
                notify("先前项目已保存；当前打开的项目与草稿保持不变。");
                return;
            }
            const changed = JSON.stringify(envelope()) !== JSON.stringify(before);
            if (copy) {
                localStorage.setItem(draftKey + ":backup:" + before.id, JSON.stringify(envelope()));
                projectId = p.id;
                if (title === before.title) title = p.title;
                $("[data-project-title]").value = title;
            }
            revision = saved.revision;
            // Keep edits made while the request was in flight, with the new CAS revision.
            draft();
            syncProjectURL();
            localStorage.setItem("t8director.lastProject", projectId);
            $("[data-save]").textContent = changed ? `已保存请求时的版本 ${revision} · 后续编辑已保留在草稿，请再次保存` : `已真实保存到服务端 · 版本 ${revision}`;
            notify(changed ? "保存期间的新编辑已保留，尚未提交到服务端；可继续编辑或再次保存。" : "项目与服务端素材身份已保存。刷新或重启后可以恢复；没有提交生成。");
            parent.postMessage({ type: "t8-director:saved", project: savedProject }, location.origin);
        } catch (e) {
            if (epoch === projectEpoch) $("[data-save]").textContent = "保存未完成 · 草稿保留 · 可重试／另存副本";
            notify(e.message); // A conflict never auto-merges or replaces either draft.
        } finally { busy = false; }
    }
    async function upload(file) {
        if (!file) throw Error("没有选择素材");
        if (navigator.onLine === false) throw Error("当前离线，素材不会提交；请恢复连接后重试，草稿仍保留");
        if (Number.isFinite(file.size) && file.size > 1024 * 1024 * 1024) {
            throw Error("单素材超过1GiB，未上传；请分段或压缩后重试");
        }
        const form = new FormData(); form.append("file", file);
        uploadNotice("正在上传 " + file.name + "（0%），成功确认后才加入本镜；失败可重试。", true);
        // Fetch does not expose upload progress in browsers.  XHR is used only
        // for this one POST so a 1 GiB boundary is visible to a beginner and a
        // disconnect never leaves a half-registered asset in the project.
        const data = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            activeUpload = xhr;
            xhr.open("POST", new URL("assets", base), true);
            xhr.withCredentials = true;
            xhr.upload.onprogress = event => {
                if (!event.lengthComputable) return;
                uploadNotice("正在上传 " + file.name + "（" + Math.round(event.loaded / event.total * 100) + "%），成功确认后才加入本镜；失败可重试。", true);
            };
            xhr.onerror = () => reject(Error("素材上传中断，服务端未确认注册；请保持草稿并重试"));
            xhr.ontimeout = () => reject(Error("素材上传超时，服务端未确认注册；请保持草稿并重试"));
            xhr.onabort = () => reject(Error("已取消素材上传；服务端未确认注册，草稿与原素材不受影响"));
            xhr.onload = async () => {
                let value = {};
                try { value = JSON.parse(xhr.responseText || "{}"); } catch { reject(Error("素材上传返回无法读取，请重试")); return; }
                if (xhr.status < 200 || xhr.status >= 300) { reject(Error(value.error || "素材上传失败，请重试")); return; }
                resolve(value);
            };
            xhr.onloadend = () => { if (activeUpload === xhr) activeUpload = null; };
            xhr.send(form);
        });
        uploadNotice("素材已由服务端确认注册。", false);
        return materialize(data);
    }
    async function compile() {
        const p = envelope();
        const epoch = projectEpoch, sequence = ++compileRequest, snapshot = JSON.stringify(p);
        const result = await request("compile", { project: p, shot_id: p.current });
        // Changing project, shot or draft invalidates this preview, not the new context.
        if (epoch !== projectEpoch || sequence !== compileRequest || snapshot !== JSON.stringify(envelope())) return result;
        const shot = result.shots?.find(s => s.id === p.current);
        if (!shot?.canvas) throw Error("预检响应缺少请求镜头的画布信息，请重试。");
        latest = result;
        const inputs = shot.canvas.preprocessing.filter(p => p.state === "cpu_processed").map(p => `<figure><img style="max-height:300px;width:100%;object-fit:contain" src="${new URL(`assets/${p.asset_id}/input-preview?width=${shot.canvas.width}&height=${shot.canvas.height}`, base)}" alt="实际CPU首尾输入"><figcaption>实际首尾输入 ${shot.canvas.width}×${shot.canvas.height} · 等比缩放${p.padding_needed ? "＋黑边填充" : ""}，不拉伸、不裁人物</figcaption></figure>`).join("");
        showDialog("当前镜头编译预检", `<p>${latest.ready ? "准备检查通过" : "请先解决当前镜头的问题"} · 编译只证明项目与素材合同，不证明画质或模型兼容。</p>${inputs}<p>其他镜头的未完成草稿不会阻止当前镜头。</p><pre>${esc(JSON.stringify({ errors: latest.errors, warnings: latest.warnings, current_shot: shot, compilation_sha256: latest.compilation_sha256 }, null, 2))}</pre>`);
        notify(latest.ready ? "当前镜头预检通过，可生成这一镜。" : "当前镜头预检未通过，素材和草稿仍保留。");
        return latest;
    }
    function viewURL(item) {
        if (!item || !item.filename) return "";
        const url = new URL("/view", location.origin);
        url.searchParams.set("filename", item.filename);
        url.searchParams.set("type", item.type || "output");
        if (item.subfolder) url.searchParams.set("subfolder", item.subfolder);
        if (item.format) url.searchParams.set("format", item.format);
        return url.href;
    }
    async function watchJob(promptId, recipe, shotId, owner) {
        if (stopWatchingJob) stopWatchingJob();
        let timer = null;
        let unknownPolls = 0;
        let resolveCompletion;
        const completion = new Promise(resolve => { resolveCompletion = resolve; });
        activeJobId = promptId;
        lastJobDialog = null;
        $("[data-service=\"job-status\"]").hidden = false;
        cancelledJobId = null;
        rememberJob(promptId, recipe, shotId, owner);
        showActiveJob(false);
        const finish = (state = "cancelled") => { if (timer) clearInterval(timer); timer = null; if (stopWatchingJob === finish) { stopWatchingJob = null; $("[data-service=\"job-status\"]").hidden = true; } resolveCompletion(state); };
        stopWatchingJob = finish;
        const poll = async () => {
            try {
                const data = await request("jobs/" + encodeURIComponent(promptId));
                if (activeJobId !== promptId) return; // Ignore an in-flight response after cancellation.
                const state = $("[data-job-state]");
                if (data.state === "unknown") {
                    unknownPolls += 1;
                    if (state) state.textContent = cancelledJobId === promptId ? "Core 已收到取消请求，任务不再跟踪。" : "Core 尚未返回任务状态，继续等待注册…";
                    if (cancelledJobId === promptId || unknownPolls >= 20) {
                        finish(cancelledJobId === promptId ? "cancelled" : "unknown");
                        activeJobId = null;
                        forgetJob(promptId);
                        notify(cancelledJobId === promptId ? "已取消当前任务；不会影响其他队列任务。" : "Core 未保留该任务状态；不会重复提交，原草稿仍保留。 ");
                    }
                    return;
                }
                unknownPolls = 0;
                if (state && data.state !== "success" && data.state !== "error") {
                    const progress = data.progress?.fraction == null ? "" : ` · ${Math.round(data.progress.fraction * 100)}%`;
                    state.textContent = "Core 状态：" + data.state + progress + " · 任务 ID " + promptId;
                }
                if (data.state === "success" || data.state === "error") {
                    finish(data.state);
                    activeJobId = null;
                    forgetJob(promptId);
                    const videos = directorOutputVideos(data);
                    const belongsHere = owner === projectId && ctx.doc().shots.some(shot => shot.id === shotId);
                    if (belongsHere) {
                        const record = { project_id: owner, shot_id: shotId, prompt_id: promptId, state: data.state, outputs: data.outputs || {}, recipe };
                        if (videos.length) completedResults.set(shotId, record);
                        ctx.recordResult(record);
                        loadResults(owner, false).catch(error => notify("成片已在当前页面，跨次打开的记录暂无法同步：" + error.message));
                    }
                    const failure = (data.status?.messages || []).filter(item => item[0] === "execution_error").at(-1)?.[1];
                    const reason = failure ? `<div class="o-warn"><strong>失败节点：${esc(failure.node_type || failure.node_id || "未知")}</strong><p>${esc(failure.exception_message || "请展开任务详情查看错误")}</p></div>` : "";
                    const media = videos.map(item => {
                        const url = viewURL(item);
                        return url ? `<video controls preload="metadata" style="max-width:100%;max-height:55vh" src="${esc(url)}"></video>` : "";
                    }).join("");
                    lastJobDialog = [promptId, data.state === "success" ? "导演台任务结果" : "导演台生成失败",
                        `<p>所属项目：${esc(owner || "旧记录未标注")} · 镜头：${esc(shotId)} · ${esc(recipe || "director")}</p>${reason}${media || "<p>本次没有可播放的视频条目。</p>"}<details><summary>任务详情与原始错误</summary><pre>${esc(JSON.stringify({ prompt_id: promptId, state: data.state, status: data.status, outputs: data.outputs }, null, 2))}</pre></details>`];
                    $("[data-service=\"job-status\"]").hidden = false;
                    if (data.state === "success" && videos.length && !belongsHere) {
                        showJobDialog(promptId, "原项目任务已完成", `<p>所属项目：${esc(owner || "旧记录未标注") }；没有写入当前镜头。请打开原项目查看成片。</p>${media}`);
                        notify("原项目任务已完成；当前项目的镜头结果未改动。");
                    } else if (data.state === "success" && videos.length) {
                        if ($("[data-dialog]").open && $("[data-dialog]").dataset.jobId === promptId) $("[data-dialog]").close();
                        notify("第 " + (ctx.doc().shots.findIndex(shot => shot.id === shotId) + 1) + " 镜已生成；点镜头卡或“镜头成片”即可直接播放。");
                    } else {
                        showJobDialog(promptId, data.state === "success" ? "生成完成但未找到视频" : "导演台生成失败",
                            `<p>${data.state === "success" ? "Core 完成，但输出没有可播放视频；任务保留。" : "Core 返回失败，原任务记录已保留。"} · ${esc(recipe || "director")}</p>${reason}${media || "<p class=\"o-muted\">本次没有可播放的视频条目。</p>"}<details><summary>任务详情与原始错误</summary><pre>${esc(JSON.stringify({ prompt_id: promptId, state: data.state, status: data.status, outputs: data.outputs }, null, 2))}</pre></details>`);
                        notify(data.state === "success" ? "未找到视频输出；可从任务状态查看详情，当前编辑保留。" : "生成失败；没有自动重采样，可从任务状态查看详情，当前编辑保留。");
                    }
                }
            } catch (error) {
                // A transient browser/Core disconnect must not resubmit the job.
                if (activeJobId !== promptId) return;
                if (error.status === 404) {
                    finish("unknown");
                    activeJobId = null;
                    forgetJob(promptId);
                    notify("Core 已找不到该任务记录；不会重复提交，原草稿仍保留。 ");
                    return;
                }
                notify("生成状态暂时无法读取，任务不会重复提交：" + error.message);
            }
        };
        timer = setInterval(poll, 1500);
        await poll();
        return completion;
    }
    async function generate() {
        if (busy) { showActiveJob(); return; }
        busy = true;
        try {
            const shotId = ctx.current();
            const project = envelope();
            const data = await submitGenerate(project, shotId, Number(sessionStorage.getItem("t8director.seed") || 26091901));
            rememberJob(data.prompt_id, data.recipe, shotId, project.id);
            if (project.id === projectId && ctx.doc().shots.some(shot => shot.id === shotId)) ctx.recordResult({ project_id: project.id, shot_id: shotId, prompt_id: data.prompt_id, state: "pending", outputs: {}, recipe: data.recipe });
            notify("已提交正式 Core 任务，断线只轮询原任务，不会重复提交。");
            await watchJob(data.prompt_id, data.recipe, shotId, project.id);
        } finally { busy = false; }
    }
    async function loadModels() { return request("models"); }
    async function submitGenerate(project, shotId, seed) {
        const fingerprintBytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify({ project, shot_id: shotId, seed })));
        const fingerprint = [...new Uint8Array(fingerprintBytes)].map(value => value.toString(16).padStart(2, "0")).join("");
        const key = `t8director.request.${project.id}.${shotId}`;
        let pending = null;
        try { pending = JSON.parse(sessionStorage.getItem(key) || "null"); } catch { /* A corrupt local key must not be reused. */ }
        const requestId = pending?.fingerprint === fingerprint ? pending.request_id : id();
        try { sessionStorage.setItem(key, JSON.stringify({ fingerprint, request_id: requestId })); } catch { /* Server-side receipt still protects a submitted ID. */ }
        const data = await request("generate", { project, shot_id: shotId, seed, client_id: tab, request_id: requestId });
        try { sessionStorage.removeItem(key); } catch { /* Best effort. */ }
        return data;
    }
    async function generateAll() {
        if (busy) { showActiveJob(); return; }
        busy = true;
        const batch = envelope();
        const shots = [...batch.doc.shots];
        let completed = 0;
        try {
            const report = await request("compile", { project: batch });
            if (!report.ready) {
                const issues = report.errors.map(error => {
                    const index = shots.findIndex(shot => shot.id === error.shot_id);
                    return (index >= 0 ? `第 ${index + 1} 镜「${shots[index].name}」：` : "全片素材：") + error.message;
                });
                showDialog("全部生成前检查 · 尚未提交任务", "<p>以下问题属于对应镜头；如果只想生成当前镜头，请关闭后使用“生成当前镜头”。</p><ul>" + issues.map(text => "<li>" + esc(text) + "</li>").join("") + "</ul>");
                notify("全片尚有未完成镜头，未提交任何生成任务；可以单独生成当前镜头。");
                return;
            }
            for (const shot of shots) {
                notify(`正在按顺序准备第 ${completed + 1}/${shots.length} 镜…`);
                const data = await submitGenerate({ ...batch, current: shot.id }, shot.id, Number(sessionStorage.getItem("t8director.seed") || 26091901) + completed);
                rememberJob(data.prompt_id, data.recipe, shot.id, batch.id);
                if (batch.id === projectId && ctx.doc().shots.some(item => item.id === shot.id)) ctx.recordResult({ project_id: batch.id, shot_id: shot.id, prompt_id: data.prompt_id, state: "pending", outputs: {}, recipe: data.recipe });
                const state = await watchJob(data.prompt_id, data.recipe, shot.id, batch.id);
                if (state !== "success") {
                    notify(`第 ${completed + 1} 镜未完成，已停止后续镜头；原任务记录已保留。`);
                    return;
                }
                completed += 1;
            }
            notify(`全部 ${completed} 镜已按顺序生成完成。`);
        } finally { busy = false; }
    }
    async function exportGraph(kind) {
        const data = await request("export", { project: envelope(), shot_id: ctx.current() });
        download(kind === "workflow" ? "director-d1-preflight.workflow.json" : "director-d1-preflight.api.json", kind === "workflow" ? data.workflow : data.api_snapshot);
        notify("已导出原生 D1 CPU预检图／API快照，不是GPU生成工作流；不会排队。");
    }
    async function showCapabilities() {
        const data = await request("capabilities");
        const rows = (data.capabilities || []).map(item => {
            const state = item.state === "ready" ? "已注册入口" : "缺少入口节点";
            return `<article class="o-asset"><h3>${esc(item.label)} · ${esc(state)}</h3><p class="o-muted">${esc(item.note)}</p><small>${esc(item.entry_nodes.join(" · "))}</small><p class="o-tip">${esc(item.execution)}</p></article>`;
        }).join("");
        showDialog("D3 配套能力检查", `<p>这里列出当前 Core 已注册的正式入口；“已注册入口”不等于 GPU 成片或人审通过。点击后仍需进入对应原生工作流，避免把不同状态合同强行拼成一个万能按钮。</p><div class="o-grid">${rows}</div>`);
    }
    async function showD3Preflight() {
        const p = envelope(), epoch = projectEpoch, sequence = ++d3Request, snapshot = JSON.stringify(p);
        const data = await request("d3/preflight", { project: p, shot_id: p.current });
        if (epoch !== projectEpoch || sequence !== d3Request || snapshot !== JSON.stringify(envelope())) return;
        const rows = (data.capabilities || []).map(item => {
            const ready = item.state === "ready_for_native_workflow";
            const deps = (item.dependencies || []).map(dep => `${dep.ok ? "✓" : "!"} ${dep.detail}`).join("；");
            const files = (item.workflows || []).map(file => `<div class="o-row o-between"><code>${esc(file)}</code><button data-d3-handoff="${esc(item.id)}" data-d3-file="${esc(file)}">${file.toLowerCase().endsWith(".json") ? "下载工作流" : "下载说明"}</button></div>`).join("");
            const contract = item.contract || {};
            const steps = (contract.steps || []).map((step, index) => (index + 1) + ". " + step).join("；");
            const requires = (contract.requires || []).join("、");
            return `<article class="o-asset"><h3>${esc(item.label)} · ${ready ? "可交接原生路线" : "需先补依赖"}</h3><p>${esc(deps)}</p><p class="o-tip">继续步骤：${esc(steps || item.next_action)}</p><p class="o-tip">需要：${esc(requires || "由原生入口继续检查")}</p><div class="o-grid">${files}</div><div class="o-row o-gap"><button data-d3-package="${esc(item.id)}">导出当前镜头路线包</button></div><p class="o-tip">${esc(contract.boundary || item.next_action)}</p></article>`;
        }).join("");
        showDialog("当前镜头 · D3 逐项预检", `<p>镜头 ${esc(p.current)} 已按服务端 project/media_map 编译。这里的“可交接”只表示入口、依赖和项目合同通过，不会偷偷排队，也不代表 GPU 或人审通过。</p><div class="o-grid">${rows}</div><pre>${esc(JSON.stringify({ schema: data.schema, compile: data.compile, warning: data.warning }, null, 2))}</pre>`);
    }
    async function compileD3() {
        const data = await request("d3/compile", { project: envelope(), shot_id: ctx.current(), seed: Number(sessionStorage.getItem("t8director.seed") || 26091901) });
        const types = Object.entries(data.nodes || {}).map(([id, node]) => id + ": " + node.class_type).join("\n");
        showDialog("D3 原生图已编译 · 未排队", "<p>路线：" + esc((data.d3_routes || []).join("、") || "默认原生") + "</p><p class=\"o-tip\">这里只证明节点图和依赖可以编译，不会加载模型、不提交队列，也不代表 GPU 或感知质量通过。</p><pre>" + esc(types + "\n\n" + JSON.stringify({ recipe: data.recipe, warning: data.warning }, null, 2)) + "</pre>");
    }
    async function handoffD3(capability, file) {
        const data = await request("d3/handoff", { capability, file });
        downloadBytes(data.name, data.text ?? JSON.stringify(data.content, null, 2));
        notify("已下载原生路线副本：" + data.name + "；不会自动映射当前镜头或排队。");
    }
    async function packageD3(capability) {
        const data = await request("d3/package", { capability, project: envelope(), shot_id: ctx.current() });
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        downloadBytes("director-" + capability + "-route-package-" + stamp + ".json", JSON.stringify(data, null, 2));
        notify("已导出 " + capability + " 路线包：包含当前项目快照、原生工作流和继续步骤；不会上传媒体或排队。");
    }
    async function openList() {
        const data = await request("projects");
        showDialog("打开服务端项目", `<p>先保存当前稿；载入会替换当前编辑页，但不删除任何素材或作品。</p><div class="o-grid">${data.projects.map(p => `<button data-open-project="${esc(p.id)}" ${p.error ? "disabled" : ""}>${esc(p.title || p.error)} · v${p.revision ?? "?"}</button>`).join("") || "还没有服务端项目，请先保存。"}</div>`);
    }
    function preserveUnknown(value, bytes, name) {
        importFile = { bytes, name };
        showDialog("未知工作流 · 原样保留／只读", `<p>这不是导演台 project.json，不会猜测或覆盖镜头。可下载原文件，再返回 ComfyUI 画布打开。</p><button data-service="original">下载原图（不改写）</button><pre>${esc(JSON.stringify(value, null, 2))}</pre>`);
    }
    $(".o-project .o-row").insertAdjacentHTML("afterbegin", '<button data-action="model-settings" aria-haspopup="dialog">模型设置</button><button data-service="save">保存项目</button><button data-service="copy">另存副本</button><button data-service="open">打开项目</button><button data-service="project">导出项目</button><button data-service="import">导入JSON</button>');
    $(".o-project h3").outerHTML = `<label>项目名称<input data-project-title aria-label="项目名称" value="${esc(title)}"></label>`;
    $(".o-footer .o-row").insertAdjacentHTML("beforeend", '<button data-service="capabilities">D3能力检查</button><button data-service="d3-preflight">当前镜头D3预检</button><button data-service="d3-compile">编译D3图</button><button data-service="workflow">导出预检工作流</button><button data-service="api">导出API快照</button>');
    $(".o-footer .o-row").insertAdjacentHTML("beforeend", '<button data-service="job-status" hidden>查看进行中任务</button>');
    $(".o-project .o-row").insertAdjacentHTML("beforeend", '<button data-service="reconnect" title="重新关联缺失的图片、视频或录音；不会删除文件或成品">重连素材</button>');
    const importer = document.createElement("input"); importer.type = "file"; importer.accept = ".json"; importer.hidden = true; ctx.root.append(importer);
    importer.onchange = async () => {
        const ticket = beginProjectLoad();
        try {
            const file = importer.files[0];
            if (!file) return;
            const bytes = new Uint8Array(await file.arrayBuffer());
            if (!acceptProjectLoad(ticket)) return;
            let p = JSON.parse(new TextDecoder().decode(bytes));
            if (p.schema !== "t8.minimax_h3.director_project") { preserveUnknown(p, bytes, file.name); return; }
            p = (await request("validate", { project: p })).project; // One authoritative lossless migration.
            if (!acceptProjectLoad(ticket)) return;
            await request("compile", { project: p }); // Validation, including versions; missing media may remain draft.
            if (!acceptProjectLoad(ticket)) return;
            retainCurrentDraft();
            hydrate(p); draft(); notify("项目已导入草稿，尚未覆盖服务端；保存时检查版本冲突。");
        } catch (e) { notify("导入失败，当前项目保留：" + e.message); }
    };
    ctx.root.addEventListener("input", e => { editEpoch++; if (e.target.matches("[data-project-title]")) { title = e.target.value; draft(); } else if (e.target.matches("[data-global],[data-field],[data-event]")) draft(); });
    ctx.root.addEventListener("click", async e => {
        const b = e.target.closest("button"); if (!b || b.disabled) return;
        const action = b.dataset.service;
        if (!["check", "generate", "generate-all", "prompts"].includes(b.dataset.action) && !action && !b.dataset.openProject && !b.dataset.reconnectAsset && !b.dataset.d3Handoff && !b.dataset.d3Package) return;
        e.preventDefault(); e.stopImmediatePropagation();
        try {
            if (b.dataset.action === "generate") await generate();
            else if (b.dataset.action === "generate-all") await generateAll();
            else if (["check", "prompts"].includes(b.dataset.action)) await compile();
            else if (b.dataset.openProject) {
                if (!confirm("载入将替换本页草稿。未保存内容可先导出项目，是否继续？")) return;
                const ticket = beginProjectLoad();
                const p = await request("projects/" + b.dataset.openProject);
                if (!acceptProjectLoad(ticket)) return;
                retainCurrentDraft();
                hydrate(p); $("[data-dialog]").close(); draft();
            } else if (b.dataset.reconnectAsset) { reconnectId = b.dataset.reconnectAsset; reconnectTarget = { id: reconnectId, context: contextToken() }; reconnectPicker.value = ""; reconnectPicker.click(); }
            else if (action === "save" || action === "copy") await save(action === "copy");
            else if (action === "job-status") showActiveJob();
            else if (action === "cancel-upload") {
                if (activeUpload) activeUpload.abort();
                else uploadNotice("当前没有正在上传的素材。", false);
            }
            else if (action === "cancel-job") {
                if (!activeJobId) { notify("当前没有可取消的导演台任务。"); return; }
                const id = activeJobId;
                const watcher = stopWatchingJob;
                cancelledJobId = id;
                const result = await request("jobs/" + encodeURIComponent(id) + "/cancel", {});
                if (activeJobId !== id || stopWatchingJob !== watcher) return;
                if (result.deleted_from_queue || result.interrupted) {
                    stopWatchingJob?.();
                    activeJobId = null;
                    forgetJob(id);
                    notify("已请求取消当前任务；不会影响其他队列任务。");
                } else notify("任务已不在队列，保留历史状态。");
            }
            else if (action === "open") await openList();
            else if (action === "capabilities") await showCapabilities();
            else if (action === "d3-preflight") await showD3Preflight();
            else if (action === "d3-compile") await compileD3();
            else if (b.dataset.d3Handoff) await handoffD3(b.dataset.d3Handoff, b.dataset.d3File);
            else if (b.dataset.d3Package) await packageD3(b.dataset.d3Package);
            else if (action === "project") download("director.project.json", envelope());
            else if (action === "workflow" || action === "api") await exportGraph(action);
            else if (action === "import") { importer.value = ""; importer.click(); }
            else if (action === "original") downloadBytes(importFile.name, importFile.bytes);
            else if (action === "reconnect") showDialog("显式重连素材", `<p>请选择原素材对应的卡片再上传替代文件，全部镜头引用会同步到新身份；不覆盖旧文件，可撤销。</p><div class="o-grid">${[...ctx.assets().values()].map(a => `<button data-reconnect-asset="${a.id}">${esc(a.name)} · ${a.id.slice(0,8)}</button>`).join("")}</div>`);
        } catch (error) { notify(error.message); if (error.data?.report) showDialog("导出未完成 · 可以修正后重试", `<pre>${esc(JSON.stringify(error.data.report.errors, null, 2))}</pre>`); }
    }, true);
    const reconnectPicker = document.createElement("input"); reconnectPicker.type = "file"; reconnectPicker.accept = "image/*,video/*,audio/*"; reconnectPicker.hidden = true; ctx.root.append(reconnectPicker);
    reconnectPicker.onchange = async () => {
        const target = reconnectTarget;
        try {
            if (!target || target.context !== contextToken()) { notify("项目已切换，未上传或重连素材。"); return; }
            const old = ctx.assets().get(target.id);
            if (!old) throw Error("原素材已不在当前项目，请重新选择。");
            const next = await upload(reconnectPicker.files[0]);
            if (target.context !== contextToken() || target !== reconnectTarget) { notify("项目或重连目标已切换；已上传素材未绑定，当前项目保持不变。"); return; }
            if (old.kind !== next.kind) throw Error("重连须同一种媒体；本次新上传仍在服务端，不覆盖原资产");
            ctx.checkpoint(); const doc = ctx.doc();
            ctx.assets().set(next.id, next);
            doc.sharedRefs = doc.sharedRefs.map(a => a === target.id ? next.id : a);
            for (const s of doc.shots) {
                for (const key of ["first", "last", "audio", "selected"]) if (s[key] === target.id) s[key] = next.id;
                for (const key of ["refs", "tray"]) s[key] = (s[key] || []).map(a => a === target.id ? next.id : a);
                if (s.audio === next.id) { s.start = 0; s.end = next.duration; }
                s.rev++;
            }
            // Keep the old library identity too: undo restores its references, not filesystem bytes.
            ctx.render(); draft(); $("[data-dialog]").close(); notify("重连完成，可撤销；原服务端文件没有被删或覆盖。");
        } catch (error) { notify(error.message); }
    };
    const restore = async () => {
        const pending = pendingJob();
        // Lock before any asynchronous project loading, not only after polling starts.
        if (pending) busy = true;
        try {
            const specific = new URL(location.href).searchParams.get("project_id");
            const local = localStorage.getItem(draftKey);
            const last = localStorage.getItem("t8director.lastProject");
            if (specific) {
                let localProject = null;
                try { localProject = local && JSON.parse(local); } catch { /* Preserve corrupt bytes below. */ }
                if (localProject?.id === specific) {
                    hydrate(localProject);
                    notify("已恢复本项目未提交草稿；保存时仍检查服务端版本，不会覆盖较新的服务端项目。");
                } else {
                    // A different URL must not silently destroy the previous tab draft.
                    if (local) localStorage.setItem(draftKey + ":backup:" + (localProject?.id || "unreadable"), local);
                    hydrate(await request("projects/" + specific)); draft();
                }
            }
            else if (local) { hydrate(JSON.parse(local)); notify("已恢复本标签未提交草稿；保存时仍检查服务端版本。"); }
            else if (last) { hydrate(await request("projects/" + last)); notify("已从服务端恢复项目与素材，不需要重传。"); }
            else draft();
        } catch (error) { notify("自动恢复未完成，请用打开项目或导入备份：" + error.message); }
        parent.postMessage({ type: "t8-director:ready" }, location.origin);
        if (pending) {
            // Resume only status polling.  A refresh must never call /generate again.
            setTimeout(() => watchJob(pending.prompt_id, pending.recipe, pending.shot_id, pending.project_id).catch(error => notify("任务恢复失败，但不会重复提交：" + error.message)).finally(() => { busy = false; }), 0);
        }
    };
    window.addEventListener("message", event => {
        if (event.origin !== location.origin || event.source !== parent || event.data?.type !== "t8-director:init") return;
        try { hydrate(event.data.project); draft(); notify("已恢复当前节点项目快照，保存前会核对服务端版本。"); }
        catch (error) { notify("节点快照未载入，原JSON保留：" + error.message); }
    });
    return { draft, upload, restore, envelope, compile, loadModels, contextToken, cancelUpload: () => activeUpload?.abort() };
}
