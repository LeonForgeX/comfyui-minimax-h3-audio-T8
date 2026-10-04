# Leon v8.37 paired Semantic Bridge source update

This is a narrowly scoped source update to LeonForgeX/comfyui-minimax-h3-audio-T8 based on paired commit `34a88c3e34e9616350105bcc95e1f90cac954483`. The existing Registry version remains 1.86.0; this does not claim a complete upstream v1.90.0 upgrade or a Registry publication.

The five Semantic Bridge implementation files are copied exactly from the official upstream T8 commit `a4762f64c802a7f321ea5b7ad1d4990db10a60f8` (2026-10-04). Existing licence and third-party notices remain applicable. No model weights or private media are included.

| File | Official upstream Git blob |
| --- | --- |
| `h3_t8/nodes_semantic_bridge.py` | `7419a56097364ece6e40701c8256948e98b31f77` |
| `h3_t8/semantic_bridge.py` | `b916a13ade6c9dfa8632ac97b2223da9c7fb76bb` |
| `h3_t8/semantic_bridge_profiles.py` | `b988e53a0bb3f4facfb4c8f9ff940161b3af58d8` |
| `h3_t8/semantic_bridge_trans.py` | `368e2aa709b6645c60cda4f83e88aa5f5b6a3826` |
| `h3_t8/semantic_bridge_mlp.py` | `af9f5d7cdbcc616a24363164f22359a68282612d` |

## Public APIs

- `MiniMaxH3SemanticBridgeConfigT8` retains its ID, input order and defaults. `chunk_tokens=0` now means the whole token sequence. Fixed application contracts are validated before sampling.
- `MiniMaxH3SemanticBridgeAutoConfigT8` reads weight content and fixed metadata contracts. It does not infer presets from filenames. Original/BUNNY six-tensor MLP uses alpha 0.10; the identified Wushu v1 content uses 0.12 and whole-sequence processing; T8 comic-combat uses its fixed alpha 1.0 contract. Unknown trainer weights require manual settings.
- `MiniMaxH3SemanticBridgeComposeT8` composes an explicit first-to-second ordered stack. The maximum is eight entries, each retains its own settings, and repeated active weight SHA256 values are rejected. This is conditioning processing rather than additive LoRA merging.
- AutoConfig and Compose are appended after the complete existing root extension list. Existing node IDs and registration positions remain unchanged.
- `MiniMaxH3SemanticBridgeApplyT8` retains single-bridge receipts. A stack returns one content-bound composite receipt with ordered per-stage receipts. Incoming already-bridged or Relay-bound conditioning is rejected; original conditioning and reference metadata are not mutated.

## Scope and validation

The existing progressive samplers, staged execution, first-pass draft API, stage residency, manifest/Accept/Context/Direct Latent authority, delivery selection, and published workflows are unchanged. Trans and trainer-exported MLP support are part of the Semantic Bridge conditioning adapter only. Kijai acceleration loaders and newer upstream sampler workflows are not included in this update.

CPU checks cover legacy numeric behavior, actual public ComfyUI V3 schemas, synthetic saved trans/MLP contracts, automatic presets, ordered composite application, immutable receipts, cancellation, reference-row preservation, cache identity, and existing native progressive/dual contracts. Test outcomes are recorded separately at the exact candidate/main commit. Synthetic CPU checks do not establish model or video quality.

GPU generation, user Windows ComfyUI-session acceptance, performance and full-video/audio human quality acceptance are `NOT_RUN`. Every new mode remains experimental; there is no universal improvement claim.
