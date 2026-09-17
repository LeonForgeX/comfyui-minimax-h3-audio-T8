# v1.84.0 — Semantic Bridge / BUNNY

Two append-only nodes, **Semantic Bridge Config** and **Semantic Bridge Apply**,
support native H3 conditioning, Prompt Relay and optional internal-loop inputs.
The previous340-node prefix, saved workflows, Sol implementation and user-selected
LoRA/Sage/Sol patch advisories are preserved; the total is342 nodes.
H16 and Meridian are not part of this release.

## 正式推荐配方 / accepted recipe

[独立双模型4+4、两段8秒工作流](../examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json)：
LOW640×320 → unchanged learned3D latent upscaler → HIGH896×448；
LOW4/HIGH4 per segment, Original/BUNNY Bridge both enabled at alpha0.10,
per_token/all_tokens, active Prompt Relay, EAV disabled, window124/context22.
The seam is near5.17s, not4s. Native joint audio and existing seam/color policies remain unchanged.

On2026-09-17 the user accepted repaired B: “可以了没问题了”.
Bound completed-media SHA256:
`6da418003510e040d2b3ccc7d9961dcb8d6b652395c0b50425f326159348ce9c`.
Four independent Bridge receipts, four completed forwards per stage, full192-frame
AV decode and byte-unchanged completed-final cache return passed. The pre-release
repair scope passed444 CPU tests and all five actual-Core graphs validated;
final release tests/package checks are recorded separately.

旧LOW448×224戏院样片及关Bridge对照仍有漂浮光斑，不作为推荐配方。
空间配置干预解决了指定样片，不等于已证明某个适配器、采样器或放大器有普遍缺陷。
Other materials, singing, reference-audio/Hybrid and arbitrary patch compositions
remain experimental; this acceptance is not a universal quality/performance guarantee.
The interrupted whole-repository diagnostic with reproduced baseline failures is
retained and is not described as all-green.

## 安装 / use

Download lossless FP16/FP32 wrappers from
[t8star/Semantic-Bridge-Comfy](https://huggingface.co/t8star/Semantic-Bridge-Comfy)
into `ComfyUI/models/semantic_bridge/t8_compat/`.
They are adapters, not LoRA or new H3 base models; no SenseNova teacher is needed.
Keep the existing H3 loaders, VAE and learned latent-upscaler dependencies.
Restart ComfyUI and refresh after updating Python; this release does not restart
the user's service or overwrite their saved workflow directory.

Use either the native optional socket or external Apply, never both on the same
conditioning. For Relay use its internal socket. Independent pass1/pass2 override
slots inherit the common config when empty; enabled=false disables only Bridge,
not the corresponding sampling stage. Changed generation settings need a new chain_id.
Unconnected/disabled/zero-strength Bridge keeps the previous route unchanged.

See [wiring and parameter guide](SEMANTIC_BRIDGE_EXP.md) and
[qualification ledger](SEMANTIC_BRIDGE_QUALIFICATION.md). GitHub publication,
formal-disk deployment, loaded user instance and Registry activation are separate states.
