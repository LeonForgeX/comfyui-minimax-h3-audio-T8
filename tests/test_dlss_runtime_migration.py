"""Legacy workflow migration is executed in Node.js, not checked by string presence."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("mode", ["static_only", "feature_probe_1_frame"])
def test_legacy_widget_migration_preserves_probe_and_devices(accepted, mode):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for JavaScript execution")
    source = Path(__file__).resolve().parents[1] / "web/dlss_runtime_migration.js"
    program = r"""
const fs = require("fs"), assert = require("assert");
let extension;
const app = {registerExtension(value) { extension = value; }};
new Function("app", fs.readFileSync(process.argv[1], "utf8").replace(/^import .*;\r?\n/, ""))(app);
class TestNode {configure(data) {this.saved = data; return 42;}}
extension.beforeRegisterNodeDef(TestNode, {name:"MiniMaxH3DLSSNRRuntimeAuditT8Advanced"});
const legacy = {widgets_values:["1.3", JSON.parse(process.argv[2]), process.argv[3], 31, 2],
 widgets_values_named:{accept_external_runtime_license: false, dxgi_adapter_index:31}};
const before = JSON.stringify(legacy), instance = new TestNode();
assert.equal(instance.configure(legacy), 42);
assert.deepEqual(instance.saved.widgets_values, ["1.3", process.argv[3], 31, 2]);
assert.equal(instance.saved.widgets_values_named.accept_external_runtime_license, undefined);
assert.equal(JSON.stringify(legacy), before);
const current = {widgets_values:["1.3", "feature_probe_1_frame", 0, 0]};
instance.configure(current); assert.strictEqual(instance.saved, current);
const malformed = {widgets_values:["1.3", false, "unknown_probe", 0, 0]};
instance.configure(malformed); assert.strictEqual(instance.saved, malformed);
class Other extends TestNode {}
const original = Other.prototype.configure;
extension.beforeRegisterNodeDef(Other, {name:"UnrelatedNode"});
assert.strictEqual(Other.prototype.configure, original);
"""
    subprocess.run([node, "-e", program, str(source), json.dumps(accepted), mode],
                   check=True, capture_output=True, text=True, timeout=30)
