"""Topaz help must remain complete, readable and workflow-neutral."""
from pathlib import Path
import shutil
import subprocess

import pytest

from h3_audio_t8_pkg import nodes_topaz


def test_regular_node_keeps_guide_out_of_execution_schema():
    schema = nodes_topaz.MiniMaxH3TopazVideoEXPT8.GET_NODE_INFO_V1()['input']
    assert 'manual_parameter_guide' not in schema['optional']
    assert 'topaz_parameter_reference' not in schema['optional']
    assert schema['required']['parameters_json'][1]['multiline'] is True
    assert '0.60' in schema['required']['vram_fraction'][1]['tooltip']
    assert '0.20' in schema['optional']['preblur'][1]['tooltip']
    assert '0.10' in schema['optional']['input_blend'][1]['tooltip']


def test_frontend_builds_complete_large_read_only_reference():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is needed for JavaScript execution')
    source = Path(__file__).resolve().parents[1] / 'web/topaz_parameter_guide.js'
    program = r"""
const fs = require('fs'), assert = require('assert');
let extension;
const app = {registerExtension(value) { extension = value; }};
global.document = {createElement(kind) {
 assert.equal(kind, 'textarea');
 const styles = {};
 return {value: '', style: {values: styles, setProperty(name, value) {styles[name] = value;}}};
}};
new Function('app', fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*;\r?\n/, ''))(app);
class TestNode {
 constructor() {this.widgets = []; this.size = [500, 300];}
 computeSize() {return [680, 1100];}
 setSize(value) {this.size = value;}
 setDirtyCanvas() {this.dirty = true;}
 addDOMWidget(name, type, inputEl, options) {
   const widget = {name, type, inputEl, options, value: inputEl.value};
   this.widgets.push(widget); return widget;
 }
 onNodeCreated() {this.created = true; return 17;}
 onConfigure() {this.configured = true; return 18;}
}
extension.beforeRegisterNodeDef(TestNode, {name: 'MiniMaxH3TopazVideoEXPT8'});
const instance = new TestNode();
assert.equal(instance.onNodeCreated(), 17);
assert.equal(instance.widgets.length, 1);
const guide = instance.widgets[0], input = guide.inputEl, text = input.value;
assert.equal(guide.name, 'topaz_parameter_reference');
assert.equal(guide.options.serialize, false);
assert.equal(input.readOnly, true);
assert.equal(input.spellcheck, false);
assert.equal(input.style.values['font-size'], '14px');
assert.equal(input.style.values['line-height'], '1.55');
assert.equal(input.style.values['overflow-y'], 'auto');
assert.equal(guide.options.getMinHeight(), 620);
assert.equal(guide.options.getHeight(), 620);
assert.deepEqual(instance.size, [760, 1100]);
assert.equal(instance.dirty, true);
for (const parameter of [
 'topaz_runtime', 'source_video', 'model_id', 'scale', 'vram_fraction',
 'parameters_json', 'size_mode', 'target_width', 'target_height',
 'output_directory', 'custom_model_id', 'output_profile', 'parameter_mode',
 'auto_estimate_frames', 'preblur', 'noise', 'details', 'halo', 'sharpen',
 'compression', 'add_noise', 'grain', 'grain_size', 'keep_color',
 'input_blend', 'engine_instances', 'gpu_device_index']) assert(text.includes(parameter), parameter);
for (let example = 1; example <= 5; example++) assert(text.includes(`${example}.`));
assert(text.includes('Topaz 软件无需保持打开'));

const legacy = new TestNode(), legacyInput = document.createElement('textarea');
legacy.widgets.push({name: 'manual_parameter_guide', value: '旧版短说明', inputEl: legacyInput, options: {}});
assert.equal(legacy.onConfigure(), 18);
assert.equal(legacy.widgets.length, 1);
assert.equal(legacy.widgets[0].value, text);
assert.equal(legacyInput.value, text);
assert.equal(legacy.widgets[0].options.serialize, false);

class Other {}
const original = Other.prototype.onNodeCreated;
extension.beforeRegisterNodeDef(Other, {name: 'OtherNode'});
assert.strictEqual(Other.prototype.onNodeCreated, original);
"""
    subprocess.run([node, '-e', program, str(source)], check=True,
                   capture_output=True, text=True, timeout=30)
