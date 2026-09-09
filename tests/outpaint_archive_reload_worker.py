"""Fresh CPU process reloads existing providers and candidate; never prepares models."""
from pathlib import Path
import json
import sys

from comfy.cli_args import args
args.cpu = True
import conftest  # noqa: E402,F401
from h3_audio_t8_pkg.video_outpaint_candidate_archive import load_selection_archive  # noqa: E402
from h3_audio_t8_pkg.video_outpaint_prepared_reload import load_prepared_outpaint  # noqa: E402


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    root, plan = Path(request["prepared_root"]), request["plan"]
    prepared, _ = load_prepared_outpaint({"plan": plan, "inspection": request["inspection"]}, root)
    handle, image = load_selection_archive(prepared, root / "candidates" / "candidate_a", request["selection_id"])
    print(json.dumps({"selection_id": handle["selection_id"], "image_shape": list(image.shape),
        "candidate_id": handle["archive_id"], "committed": len(handle["windows"].snapshot()["committed"]),
        "model_loaded": False, "sampler_called": False}))
