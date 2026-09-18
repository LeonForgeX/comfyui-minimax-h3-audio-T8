import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "minimax-h3-audio-t8.taeh3-private-preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3TAEH3SamplingPreviewEXPT8") return;
        const created = nodeType.prototype.onNodeCreated;
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            this._t8PreviewCleanup?.();
            const panel = document.createElement("div");
            panel.style.cssText = "box-sizing:border-box;height:100%;display:flex;flex-direction:column;gap:8px;padding:10px;color:var(--fg-color,#ddd);font:14px/1.5 system-ui,sans-serif";
            const image = document.createElement("img");
            image.alt = "TAEH3近似采样预览，不是最终成片";
            image.style.cssText = "width:100%;height:300px;object-fit:contain;background:#151515";
            const status = document.createElement("div");
            status.style.cssText = "white-space:pre-wrap;overflow:auto;min-height:80px";
            status.textContent = "运行后播放实际x0的开头片段。近似预览不代表最终画质／音频，不会自动排队。";
            const controls = document.createElement("div");
            const play = document.createElement("button");
            play.textContent = "暂停预览";
            const cancel = document.createElement("button");
            cancel.textContent = "取消当前绑定采样";
            cancel.disabled = true;
            controls.append(play, cancel);
            panel.append(image, status, controls);
            this.addDOMWidget("t8_taeh3_preview", "preview", panel, {
                serialize: false, getMinHeight: () => 440, getMaxHeight: () => 800,
            });
            this.setSize?.([Math.max(this.size?.[0] ?? 0, 600), Math.max(this.size?.[1] ?? 0, 780)]);
            let current = null, frames = [], index = 0, timer = null, paused = false, disposed = false;
            const stopTimer = () => { if (timer !== null) clearInterval(timer); timer = null; };
            const animate = () => {
                stopTimer();
                if (!frames.length) return;
                image.src = frames[index % frames.length];
                if (!paused && frames.length > 1) timer = setInterval(() => {
                    index = (index + 1) % frames.length;
                    image.src = frames[index];
                }, 1000 / Math.max(1, Math.min(24, current?.fps ?? 12)));
            };
            const matches = (bound) => !disposed && current?.prompt === bound.prompt && current?.run === bound.run;
            const listener = (event) => {
                const d = event.detail;
                if (disposed || String(d?.preview_node_id) !== String(this.id)) return;
                if (typeof d?.epoch_ns !== "string" || !/^\d+$/.test(d.epoch_ns) || !Number.isSafeInteger(d.sequence)) return;
                const epoch = BigInt(d.epoch_ns);
                if (d.kind === "start") {
                    if (current && epoch <= current.epoch) return;
                    current = { epoch, sequence: d.sequence, prompt: d.prompt_id, run: d.run_id, fps: d.fps, phase: d.phase, active: true };
                    frames = []; index = 0; stopTimer(); image.removeAttribute("src");
                    cancel.disabled = false;
                    status.textContent = `阶段：${d.phase}；请求：${d.prompt_id}\n开头连续片段近似预览；不是最终画质／声音。`;
                    return;
                }
                if (!current || d.prompt_id !== current.prompt || d.run_id !== current.run || epoch !== current.epoch || d.sequence <= current.sequence) return;
                current.sequence = d.sequence;
                if (d.kind === "frames") {
                    if (!Array.isArray(d.frames) || d.frames.length > 24 || !d.frames.every(f => typeof f === "string" && f.startsWith("data:image/jpeg;base64,") && f.length < 1500000) || d.frames.reduce((sum, f) => sum + f.length, 0) > 1500000) return;
                    frames = d.frames; index = 0; animate();
                    const coverage = d.decoder_family === "independent_2d_latent_frames"
                        ? `2D逐潜帧预览：开头 ${d.decoded_prefix_frames} 张；不是连续24fps视频，播放速度不是源时间。`
                        : `时序TAEH3：仅开头 ${d.decoded_prefix_frames} 帧（原生24fps）。`;
                    status.textContent = `阶段：${d.phase}；实际步 ${d.step}/${d.steps}${d.global_step != null ? `；本段累计步 ${d.global_step}/${d.global_total}` : ""}\n${coverage} 预览播放${current.fps}fps；可能缩小，不代表最终细节。\n请求：${current.prompt}`;
                } else if (d.kind === "unavailable") {
                    status.textContent = `预览不可用，原采样继续：${d.reason}\n请求：${current.prompt}`;
                } else if (d.kind === "end") {
                    current.active = false;
                    cancel.disabled = true;
                    status.textContent += `\n此阶段结束：${d.status}；${d.coverage}。留存画面属于上面请求，不是下一次实时预览。`;
                }
            };
            const onPlay = () => { paused = !paused; play.textContent = paused ? "播放预览" : "暂停预览"; animate(); };
            const onCancel = async () => {
                if (!current || cancel.disabled) return;
                const bound = { prompt: current.prompt, run: current.run };
                if (typeof bound.prompt !== "string" || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(bound.prompt)) {
                    status.textContent += "\n无法安全绑定请求；请使用ComfyUI原生取消按钮。";
                    return;
                }
                cancel.disabled = true;
                try {
                    const response = await api.fetchApi(`/api/jobs/${encodeURIComponent(bound.prompt)}/cancel`, { method: "POST" });
                    const data = response.ok ? await response.json() : null;
                    if (!matches(bound)) return;
                    status.textContent += response.ok ? (data?.cancelled ? "\n已发送本请求取消，等待采样器确认。" : "\n此请求已结束或未在队列中；未取消其他任务。") : (response.status === 404 ? "\n本Core没有可用的按请求取消接口；请使用原生取消按钮。不会调用全局interrupt。" : `\n取消接口返回HTTP${response.status}，尚未确认中断；可重试或使用原生取消。不会调用全局interrupt。`);
                    if (!response.ok && response.status !== 404 && current.active) cancel.disabled = false;
                } catch (error) {
                    if (matches(bound)) {
                        status.textContent += `\n取消发送失败：${error.message}；未操作其他任务。`;
                        if (current.active) cancel.disabled = false;
                    }
                }
            };
            const stopPropagation = e => e.stopPropagation();
            panel.addEventListener("pointerdown", stopPropagation);
            panel.addEventListener("wheel", stopPropagation);
            play.addEventListener("click", onPlay);
            cancel.addEventListener("click", onCancel);
            api.addEventListener("t8-taeh3-sampling-preview", listener);
            this._t8PreviewCleanup = () => {
                disposed = true; stopTimer(); frames = []; current = null;
                api.removeEventListener("t8-taeh3-sampling-preview", listener);
                play.removeEventListener("click", onPlay); cancel.removeEventListener("click", onCancel);
                panel.removeEventListener("pointerdown", stopPropagation); panel.removeEventListener("wheel", stopPropagation);
                image.removeAttribute("src"); panel.remove(); this._t8PreviewCleanup = null;
            };
            return result;
        };
        nodeType.prototype.onRemoved = function () {
            this._t8PreviewCleanup?.();
            return removed?.apply(this, arguments);
        };
    },
});
