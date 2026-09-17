from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import tomllib

import h3_audio_t8_pkg
from h3_audio_t8_pkg.sol_attn_minimax_v2 import SolAttnMiniMax


ROOT = Path(__file__).resolve().parents[1]
SOL_SHA256 = "931c3602d7433a1dad313aebbbbe13067fdf6c1c607cdcb1b52ac173ae21e5e6"


def test_original_sol_source_is_registered_once_without_adapter_or_edits():
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [cls.define_schema().node_id for cls in classes]
    assert len(ids) == len(set(ids)) == 340
    assert ids[336:339] == [
        "MiniMaxH3FastH3V2SetupEXPT8",
        "MiniMaxH3FastH3V2RuntimeAuditEXPT8",
        "MiniMaxH3FastH3V2DualModelLongVideoEXPT8",
    ]
    assert classes[-1] is SolAttnMiniMax
    assert ids[-1] == "SolAttnMiniMax"
    source = Path(inspect.getfile(SolAttnMiniMax))
    assert source.resolve() == (ROOT / "sol_attn_minimax_v2.py").resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOL_SHA256
    assert ids == json.loads((ROOT / "features.json").read_text(encoding="utf-8"))["nodes"]


def test_sol_original_schema_defaults_and_distribution_inclusion():
    schema = SolAttnMiniMax.define_schema()
    assert schema.display_name == "Patch Sol-Attn (MiniMax)"
    assert schema.category == "sol_attn"
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["tau"].default == 1.3
    assert inputs["min_tokens"].default == 12288
    assert inputs["sink_conditioning"].default == "exact_kv_and_rows"
    assert inputs["morton"].default is False
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "sol_attn_minimax_v2.py" in config["tool"]["comfy"]["includes"]
