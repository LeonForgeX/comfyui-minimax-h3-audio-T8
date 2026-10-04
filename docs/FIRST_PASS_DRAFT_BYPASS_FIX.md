# Native Bypass LoRA first-pass draft fix

This source-only Leon fork update pairs with Leon v8.38.1. It leaves upstream Registry version 1.86.0, node schemas, workflows and sampling defaults unchanged.

## Fixed failure

With first-pass draft saving enabled, native ComfyUI Bypass LoRA could finish LOW and then fail before HIGH with `First-pass draft execution inputs changed`. Sampled MODEL clones share their network. Core installs native bypass forwards when loading a clone and restores native bound forwards when ejecting it. The old draft identity treated those normal lifecycle changes as an input change. LOW and HIGH using different injections on a shared network also exposed this error.

The fix recognizes only authenticated Core bypass injection factories and their exact live owners. Identity still binds each selected stage's adapter, strength, configuration and effective compute tensor bytes. A narrow check observes the installed LOW target forwards before Core teardown; full input and identity validation remains after stage-owned KJ/Relay cleanup. Real weight, configuration, forward-owner and input changes still fail. NaN, kernel errors and cancellation still propagate.

## Usage

Update both this Leon T8 fork and Leon to the paired version, fully restart ComfyUI, and refresh the browser. Run one complete generation to establish a draft with the new implementation identity. Older implementation drafts require a fresh first pass.

- An unaudited LOW injection remains nonportable: generation continues, but LOW is not saved for reuse across executions. The compatibility advisory can still appear.
- HIGH-only native Bypass LoRA does not make an otherwise authenticated LOW nonportable. That LOW draft can be reused when changing HIGH LoRA or the independent HIGH seed.
- Comfy model compiler graph-break counts are informational and are separate from this draft-boundary failure. This fix does not certify arbitrary compiler or third-party patch stacks.

No sampling arithmetic, steps, sigmas, LoRA strength, audio or geometry settings were changed.

## Validation

The previous exact paired T8 main reproduced four failures at the draft save boundary; draft-disabled controls completed LOW and HIGH. The final fix passed 25 focused regression cases and the existing native CPU/policy suite (537 passed, 22 skipped). Cases cover separate/shared LOW and HIGH injections, native inject/eject and dtype casts, mutation/error/cancellation checks, HIGH-only portable LOW replay with a changed HIGH seed, and actual KJ stage ownership with bypass.

Tests use pinned ComfyUI Core, a tiny native H3/Euler CPU model, a labelled lifter double, and a CPU SDPA double for the optional KJ Sage kernel. No full GPU model generation, visual-quality, speed or VRAM claim is made by these tests. Existing source workflows and Registry publication settings are unchanged.
