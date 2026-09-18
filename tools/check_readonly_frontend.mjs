// Evaluate the actual shipped extensions against a bounded fake DOM/API.
// No browser/network/user frontend/generation; optional fresh artifact receipt only.
import assert from "node:assert/strict";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.listeners = new Map(); this.style = {}; }
    append(...children) { this.children.push(...children); }
    addEventListener(name, callback) { this.listeners.set(name, callback); }
    removeEventListener(name, callback) { if (this.listeners.get(name) === callback) this.listeners.delete(name); }
    setAttribute(name, value) { this[name] = value; }
    removeAttribute(name) { delete this[name]; }
    remove() { this.removed = true; }
}
let extension, pending;
const listeners = new Map(), timers = new Map(), requests = [];
const api = {
    addEventListener: (name, callback) => listeners.set(name, callback),
    removeEventListener: (name, callback) => { if (listeners.get(name) === callback) listeners.delete(name); },
    fetchApi: (url, options) => { requests.push({ url, options }); return new Promise(resolve => { pending = resolve; }); },
};
const context = vm.createContext({
    app: { registerExtension: value => { extension = value; } }, api,
    document: { createElement: name => new Element(name) },
    setInterval: callback => { const id = timers.size + 1; timers.set(id, callback); return id; },
    clearInterval: id => timers.delete(id),
});
function load(name) {
    const path = fileURLToPath(new URL(`../web/${name}`, import.meta.url));
    const source = readFileSync(path, "utf8").replace(/^import .*;\r?\n/gm, "");
    vm.runInContext(source, context, { filename: path });
}
function nodeClass() {
    return class Node {
        constructor() { this.id = 12; this.size = [200, 100]; this.created = this.removed = this.executed = 0; }
        onNodeCreated() { this.created++; return "prior-created"; }
        onRemoved() { this.removed++; return "prior-removed"; }
        onExecuted() { this.executed++; return "prior-executed"; }
        addDOMWidget(name, type, element, options) { this.widget = { name, type, element, options }; }
        setSize(size) { this.size = size; }
    };
}
const A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
load("taeh3_preview.js");
const Preview = nodeClass();
await extension.beforeRegisterNodeDef(Preview, { name: "MiniMaxH3TAEH3SamplingPreviewEXPT8" });
const node = new Preview();
assert.equal(node.onNodeCreated(), "prior-created");
assert.equal(node.created, 1);
assert.equal(node.widget.options.serialize, false);
assert.ok(node.size[0] >= 600 && node.size[1] >= 780);
const [image, status, controls] = node.widget.element.children;
const [play, cancel] = controls.children;
const emit = d => listeners.get("t8-taeh3-sampling-preview")({ detail: { preview_node_id: "12", ...d } });
emit({ kind: "start", preview_node_id: "other", epoch_ns: "10", sequence: 1, prompt_id: A, run_id: "a" });
assert.equal(cancel.disabled, true);
const startA = { kind: "start", epoch_ns: "10", sequence: 1, prompt_id: A, run_id: "a", phase: "low", fps: 12 };
emit(startA);
const frame = { ...startA, kind: "frames", sequence: 2, frames: ["data:image/jpeg;base64,YQ==", "data:image/jpeg;base64,Yg=="], step: 2, steps: 4, decoded_prefix_frames: 22 };
emit(frame);
assert.equal(image.src, frame.frames[0]);
assert.equal(timers.size, 1);
assert.ok(status.textContent.includes("时序TAEH3") && status.textContent.includes("原生24fps"));
emit({ ...frame, sequence: 3, decoder_family: "independent_2d_latent_frames", source_prefix_fps: null, decoded_prefix_frames: 7 });
assert.ok(status.textContent.includes("2D逐潜帧预览") && status.textContent.includes("不是连续24fps视频"));
assert.ok(!status.textContent.includes("原生24fps"));
emit({ ...frame, sequence: 1, frames: ["data:image/jpeg;base64,old"] });
assert.equal(image.src, frame.frames[0]);
const cancellingA = cancel.listeners.get("click")();
assert.equal(requests.length, 1);
assert.equal(requests[0].url, `/api/jobs/${A}/cancel`);
assert.equal(requests[0].options.method, "POST");
emit({ ...startA, epoch_ns: "20", prompt_id: B, run_id: "b" });
const beforeB = status.textContent;
pending({ ok: true, json: async () => ({ cancelled: true }) });
await cancellingA;
assert.equal(status.textContent, beforeB);
assert.equal(image.src, undefined);
assert.equal(timers.size, 0);
emit({ ...frame, kind: "end", sequence: 99, active: false, status: "finished" });
assert.equal(cancel.disabled, false); // stale A cannot disable B cancellation
const cancellingB = cancel.listeners.get("click")();
pending({ ok: false, status: 404 });
await cancellingB;
assert.ok(status.textContent.includes("不会调用全局interrupt"));
assert.equal(requests.length, 2);
assert.ok(requests.every(r => !r.url.includes("interrupt")));
emit({ ...startA, epoch_ns: "30", prompt_id: B, run_id: "c" });
const transientFailure = cancel.listeners.get("click")();
pending({ ok: false, status: 503 });
await transientFailure;
assert.equal(cancel.disabled, false);
assert.ok(status.textContent.includes("HTTP503"));
const endingDuringRetry = cancel.listeners.get("click")();
emit({ ...startA, kind: "end", epoch_ns: "30", prompt_id: B, run_id: "c", sequence: 2, status: "cancelled" });
pending({ ok: false, status: 503 });
await endingDuringRetry;
assert.equal(cancel.disabled, true); // a failed reply must not reactivate an ended request
emit({ ...startA, epoch_ns: "40", prompt_id: A, run_id: "d" });
emit({ ...frame, epoch_ns: "40", prompt_id: A, run_id: "d",
    frames: Array(24).fill("data:image/jpeg;base64," + "YQ==".repeat(17500)) });
assert.equal(image.src, undefined); // cumulative payload bound, not only per-image bound
assert.equal(timers.size, 0);
play.listeners.get("click")();
assert.equal(requests.length, 4);
assert.equal(node.onRemoved(), "prior-removed");
assert.equal(node.removed, 1);
assert.equal(listeners.size, 0);
assert.equal(timers.size, 0);
assert.equal(image.src, undefined);

load("readable_audio.js");
for (const name of ["MiniMaxH3NodeSourceDiagnosticT8", "MiniMaxH3AudioSourceExplanationT8"]) {
    const Readable = nodeClass();
    await extension.beforeRegisterNodeDef(Readable, { name });
    const report = new Readable();
    report.onNodeCreated();
    assert.equal(report.widget.element.readOnly, true);
    assert.equal(report.widget.options.serialize, false);
    assert.equal(report.onExecuted({ text: ["<script>untrusted report</script>\nlong text"] }), "prior-executed");
    assert.equal(report.widget.element.value, "<script>untrusted report</script>\nlong text");
    assert.equal(report.widget.element.innerHTML, undefined);
    assert.ok(report.size[0] >= 560 && report.size[1] >= 380);
    report.onRemoved();
    assert.equal(report.widget.element.listeners.size, 0);
}
const result = { status: "actual_extension_fake_DOM_pass", browser_used: false, network_used: false,
    checked: ["callback_chaining", "private_node_routing", "monotonic_events", "stale_cancel_reply", "request_scoped_cancel", "no_global_interrupt", "bounded_animation_cleanup", "readonly_diagnostic_text", "transient_cancel_retry", "ended_request_not_reactivated", "cumulative_payload_bound", "2D_latent_frames_not_reconstructed_24fps", "temporal_TAEHV_label_preserved"],
    source_hashes: Object.fromEntries(["taeh3_preview.js", "readable_audio.js"].map(name =>
        [name, createHash("sha256").update(readFileSync(fileURLToPath(new URL(`../web/${name}`, import.meta.url)))).digest("hex")])) };
if (process.argv[2] === "--receipt") {
    const project = fileURLToPath(new URL("../", import.meta.url));
    const path = resolve(process.argv[3]);
    const inside = relative(resolve(project, "artifacts/five-track-development-20260918"), path);
    assert.ok(inside && !inside.startsWith("..") && !inside.includes(":"));
    assert.equal(existsSync(path), false);
    writeFileSync(path, JSON.stringify(result, null, 2), { flag: "wx" });
}
console.log(JSON.stringify(result));
