"""CPU actual-RGB source preservation through the production outpaint compositor."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file,write_new_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    evidence = json.loads((root/"independent-special-audit.json").read_text(encoding="utf8"))
    if evidence["status"] != "actual_core_and_trt_standard_bounded_short_decode_verified_not_human":
        raise ValueError("Actual special-interface RGB must be independently audited first")
    if digest_file(root/"outputs.safetensors") != evidence["files"]["outputs.safetensors"]:
        raise ValueError("Actual RGB changed")
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    package = types.ModuleType("t8_trt_anchor_audit")
    package.__path__ = [str(PROJECT / "h3_t8"), str(PROJECT)]
    sys.modules[package.__name__] = package
    build = importlib.import_module(package.__name__+".video_outpaint_plan").build_outpaint_plan
    composite = importlib.import_module(package.__name__+".video_outpaint_composite").composite_outpaint_frames
    plan = build(source_sha256=digest_file(root/"outputs.safetensors"),width=768,height=384,frame_count=39,
                 aspect="custom",left=128,right=128,top=64,bottom=64,generation_megapixels=0,window_frames=39)
    rows = []
    with safe_open(root/"outputs.safetensors",framework="pt",device="cpu") as f:
        native,trt = f.get_slice("native"),f.get_slice("trt_bounded")
        for index in range(39):
            # A source rectangle from the real native video, not a synthetic RGB.
            source = native[0,:,index:index+1,64:448,128:896].movedim(0,-1).contiguous()
            candidate = trt[0,:,index:index+1].movedim(0,-1).contiguous()
            before = candidate.clone()
            source_before = source.clone()
            result,receipt = composite(source,candidate,plan,start_frame=index)
            outside = torch.ones((512,1024),dtype=torch.bool)
            outside[64:448,128:896] = False
            if (not torch.equal(result[:,64:448,128:896],source) or not torch.equal(result[:,outside],candidate[:,outside])
                    or not torch.equal(candidate,before) or not torch.equal(source,source_before)
                    or not receipt["source_exact_before_encoding"]):
                raise ValueError("Source RGB, generated borders or immutable input contract violated")
            rows.append(receipt)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU pixel-anchor audit initialized CUDA")
    report = {"status":"actual_trt_rgb_compositor_pixel_preservation_verified","frames":39,
              "source_rectangle":[128,64,896,448],"all_source_pixels_exact":True,"all_outside_pixels_unchanged":True,
              "inputs_unchanged":True,"cuda_initialized":False,"receipts":rows,
              "sources":{str(p):digest_file(p) for p in (Path(__file__),PROJECT/'h3_t8/video_outpaint_plan.py',PROJECT/'h3_t8/video_outpaint_composite.py',root/"outputs.safetensors",root/"independent-special-audit.json")},
              "limits":"CPU production compositor tested on actual native/TRT39-frame RGB with literal source rectangle. No newly generated outpainting, seam-quality or lossy-MP4 pixel equality claim."}
    write_new_json(root/"pixel-anchor-audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("receipts","sources")},indent=2))


if __name__ == "__main__":
    main()
