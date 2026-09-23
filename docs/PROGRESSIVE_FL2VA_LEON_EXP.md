# Progressive single-clip FL2VA experiment

The public `MiniMaxH3ProgressiveSamplerEXPT8` node now accepts `task=fl2va` for an **empty native AV template**. It requires exactly two positive conditioning keyframes, at native frame indices `0` and `frame_count - 1`. Both native latents must match the final output canvas, have finite floating-point values and the normal H3 batch/channel shape. LOW resizes each reference through the existing selected guide policy; HIGH uses the originals. The node returns the final AV and its existing sampling report. This does not accept initialized AV, masks, extra references or long-video continuation under `fl2va`.

Leon can use this route for an independent FL2V Director material pack while keeping candidate Review and Accept in Leon. It needs a compatible FL2VA model family, an installed learned upscaler and enough GPU memory for both stages. T2VA and I2VA contracts retain their original behavior. No T8 package version, Registry publication or upstream branch is changed by this experimental source branch.

The CPU tests in `tests/test_progressive_fl2va_public.py` cover two keyframe latents, their LOW/HIGH shape and no mutation, wrong temporal indices, invalid latent geometry and non-finite values. They do not prove Windows H3 generation, video quality, or peak GPU memory.

User-selected hooks and MODEL patches follow the existing advisory policy: preserve them and report unverified combinations. Additional reference media, malformed keyframes and invalid native AV still raise input errors.
