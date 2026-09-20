import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BASE = "/minimax_h3_t8/director";
let overlay;
function openDirector(node) {
    if (overlay) { overlay.close(); overlay.remove(); overlay = null; }
    const dialog = document.createElement("dialog");
    dialog.style.cssText = "position:fixed;inset:1vh 1vw;width:98vw;height:98vh;max-width:none;max-height:none;padding:0;border:1px solid #62523c;border-radius:14px;background:#101314;color:#fff;z-index:10000";
    const bar = document.createElement("div");
    bar.style.cssText = "height:42px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;font-size:14px";
    const label = document.createElement("span"); label.textContent = "曜石导演台 · D2a–D2c 真实生成";
    const close = document.createElement("button"); close.textContent = "返回画布";
    close.onclick = () => { dialog.close(); dialog.remove(); overlay = null; };
    bar.append(label, close);
    const frame = document.createElement("iframe");
    frame.title = "曜石导演台"; frame.src = api.apiURL(BASE + "/ui");
    frame.style.cssText = "width:100%;height:calc(100% - 42px);border:0;display:block";
    frame.addEventListener("load", () => { label.textContent = "曜石导演台 · 已连接当前 Core"; });
    frame.addEventListener("error", () => { label.textContent = "曜石导演台 · 页面加载失败，返回画布后可重试"; });
    dialog.append(bar, frame); document.body.append(dialog); dialog.showModal(); overlay = dialog;
    const receive = (event) => {
        if (event.origin !== location.origin || event.source !== frame.contentWindow || !node) return;
        const project = node.widgets?.find(w => w.name === "project_json");
        if (event.data?.type === "t8-director:ready") {
            if (project?.value) {
                try { frame.contentWindow.postMessage({ type: "t8-director:init", project: JSON.parse(project.value) }, location.origin); }
                catch { label.textContent = "节点项目JSON无效，原内容保留，请用项目备份恢复"; }
            }
            return;
        }
        if (event.data?.type !== "t8-director:saved") return;
        const shot = node.widgets?.find(w => w.name === "shot_id");
        if (project) project.value = JSON.stringify(event.data.project);
        if (shot) shot.value = event.data.project.current;
        app.graph?.setDirtyCanvas(true, true);
    };
    window.addEventListener("message", receive);
    dialog.addEventListener("close", () => window.removeEventListener("message", receive), { once: true });
}

app.registerExtension({
    name: "T8.ObsidianDirector.D1",
    nodeCreated(node) {
        if (node.comfyClass !== "MiniMaxH3DirectorProjectT8") return;
        node.addWidget("button", "打开曜石导演台", null, () => openDirector(node), { serialize: false });
    },
    setup() {
        const button = document.createElement("button");
        button.textContent = "T8 导演台"; button.title = "D2a–D2c · 当前镜头进入正式 Core 队列";
        button.id = "t8-director-open";
        button.style.cssText = "position:fixed;bottom:18px;right:18px;z-index:1000;padding:10px 15px;border:1px solid #7d6848;border-radius:10px;background:#1c2023;color:#e8c68e;font-size:14px;cursor:pointer";
        button.onclick = () => openDirector(); document.body.append(button);
    }
});
