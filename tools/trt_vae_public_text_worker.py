"""Fixed readable text chart through actual native/TRT VAE. No video sampling."""
import argparse
import copy
import gc
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from dlss_fi_backend.resources import SerialProbeLease  # noqa: E402


def chart(font):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (1024, 512), '#eeeeee')
    draw = ImageDraw.Draw(image)
    draw.rectangle((512, 0, 1023, 511), fill='#15212e')
    rows = [(28, 36, 'H3 VAE 2026 / 0123456789'),
            (98, 32, '中文细节：你在哪里'),
            (164, 24, 'MiniMax H3 - Edge / AaBb 8B'),
            (216, 24, '游戏任务：守卫城门 24 FPS'),
            (268, 18, 'Thin text 0123456789 ABC abc'),
            (312, 18, '中文小字：速度与画质对照'),
            (354, 14, 'Small text - Not an OCR accuracy benchmark')]
    boxes = []
    for x, color in ((16, '#101820'), (528, '#f0f0f0')):
        for y, size, text in rows:
            face = ImageFont.truetype(str(font), size)
            draw.text((x, y), text, fill=color, font=face)
            boxes.append({'text': text, 'font_size': size, 'bbox': list(draw.textbbox((x, y), text, font=face))})
    for x in range(16, 1008, 4):
        draw.line((x, 416, x, 470), fill='#d03240' if x % 8 else '#1689bb', width=1)
    return image, boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    path = parser.parse_args().request
    root, request = path.parent, json.loads(path.read_text(encoding='utf8'))
    if request['schema'] != 't8-trt-public-text1-v1' or not request['parent_holds_serial_leases']:
        raise ValueError('Use the owned short text controller')
    for name, sha in request['sources'].items():
        if digest_file(name) != sha:
            raise ValueError('Text probe source changed: ' + Path(name).name)
    sys.path.insert(0, request['core'])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import numpy as np
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() != request['gpu_uuid'].removeprefix('GPU-').lower():
        raise ValueError('GPU differs from controller')
    image, boxes = chart(request['font'])
    image.save(root / 'source.png')
    pixels = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255
    write_new_json(root / 'text-layout.json', {'rows': boxes, 'scope': 'Synthetic static text, not H3-generated writing or temporal fidelity'})
    state, metadata = comfy.utils.load_torch_file(request['native_vae'], return_metadata=True)
    native = comfy.sd.VAE(sd=state, metadata=metadata, device=torch.device('cuda:0'), dtype=torch.float16)
    del state
    native.throw_exception_if_invalid()
    counts = {'encode': 0, 'decode': 0}
    def count(kind):
        def hook(module, inputs):
            counts[kind] += 1
        return hook
    hooks = [native.first_stage_model.encoder.register_forward_pre_hook(count('encode')),
             native.first_stage_model.decoder.register_forward_pre_hook(count('decode'))]
    print('TEXT_PHASE native_encode_and_decode', flush=True)
    with torch.inference_mode():
        z = native.encode(pixels).cpu()
        reference = native.decode(z)[0].cpu()
    for hook in hooks:
        hook.remove()
    mm.unload_all_models()
    native.first_stage_model.to('cpu')
    del native
    gc.collect()
    torch.cuda.empty_cache()
    package = types.ModuleType('t8_public_text_probe')
    package.__path__ = [str(PROJECT)]
    sys.modules[package.__name__] = package
    nodes = importlib.import_module(package.__name__ + '.nodes_trt_vae')
    result = nodes.MiniMaxH3TRTVAEFullEXPT8.execute(Path(request['native_vae']).name,
        request['decoder'], request['t1'], request['encoder'], request['runtime_site'], 1024)
    vae = result.result[0]
    backend = vae.open_backend
    backend.serial_lease = lambda: SerialProbeLease(Path(tempfile.gettempdir()) / 'T8-TRT-public-text-worker.lock')
    def check():
        if (root / 'cancel.request').exists():
            raise InterruptedError('Text probe cancelled')
    backend.check = vae.check = check
    leases = []
    print('TEXT_PHASE actual_public_same_latent_and_full_roundtrip', flush=True)
    with torch.inference_mode():
        same_latent = vae.decode(z)[0].cpu()
        leases.append(copy.deepcopy(backend.last_report))
        trt_z = vae.encode(pixels)
        leases.append(copy.deepcopy(backend.last_report))
        full = vae.decode(trt_z)[0].cpu()
        leases.append(copy.deepcopy(backend.last_report))
    tensors = {'source': pixels, 'native_latent': z, 'trt_latent': trt_z,
               'native': reference, 'trt_same_latent': same_latent, 'trt_full': full}
    for key, value in tensors.items():
        expected = (1, 24, 1, 32, 64) if 'latent' in key and key != 'trt_same_latent' else (1, 512, 1024, 3)
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError('Text tensor shape/finite failure: ' + key)
    if counts != {'encode': 15, 'decode': 15} or [r['calls'] for r in leases] != [15, 15, 15]:
        raise ValueError('Expected full0.5MP T1 tile dispatch')
    if any(r['status'] != 'complete' or r['free_before_deserialize'] - r['free_after_release'] > 512*1024**2 for r in leases):
        raise ValueError('TRT execution/release gate failed')
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(root / 'outputs.safetensors'))
    from PIL import Image
    for key in ('native', 'trt_same_latent', 'trt_full'):
        Image.fromarray((tensors[key][0].clamp(0, 1).numpy()*255).round().astype(np.uint8)).save(root / (key+'.png'))
    names = ('source.png', 'native.png', 'trt_same_latent.png', 'trt_full.png', 'text-layout.json', 'outputs.safetensors')
    write_new_json(root / 'result.json', {'status': 'actual_public_text_requires_independent_audit',
        'native_calls': counts, 'leases': leases, 'files': {n: digest_file(root/n) for n in names},
        'limits': 'One synthetic0.5MP static text chart. No sampling, video/flicker, OCR score, speed or human acceptance.'})


if __name__ == '__main__':
    main()
