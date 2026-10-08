"""Explicit file-backed RGB/PCM continuation, without sampled-parent ancestry.

The native builder's segment coordinate1 is local conditioning geometry only.
No accepted manifest, original latent, MODEL, seed or sampling ancestor is made.
This API is loaded by the installed package for Director adapters; it neither
samples nor modifies source media. Unknown VAE wrappers remain nonportable.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from . import core, long_video
from .patch_stack_policy import (
    _execution_selection,
    model_identity_matches,
    nonportable_component_identity,
)

SCHEMA = 't8.external-rgb-pcm-source.v1'
CONTEXT_SCHEMA = 't8.external-rgb-pcm-context.v1'
AUDIO_POLICIES = {'video_only', 'reencode_stereo_pcm_context'}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _inside(root, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or '\\' in relative or '..' in Path(relative).parts):
        raise ValueError('External source requires a contained relative file path')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('External source file escapes its bound root')
    return path


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-fA-F]{64}', value) is None:
        raise ValueError('External source requires an actual SHA256')
    return value.lower()


def _tensor_identity(value):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())):
        raise ValueError('External context must contain finite floating tensors')
    data = value.detach().cpu().contiguous()
    return {'shape': list(data.shape), 'dtype': str(data.dtype),
            'sha256': hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()}


def _implementation():
    return {Path(module.__file__).name: _file_sha(Path(module.__file__))
            for module in (core, long_video)} | {Path(__file__).name: _file_sha(Path(__file__))}


def _interrupt():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def _ratio(value):
    return {'num': value.numerator, 'den': value.denominator}


def rounded_audio_frame_sample(frame, sample_rate):
    """Existing delivery round rule, applied to absolute source-frame endpoints."""
    if type(frame) is not int or frame < 0 or type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('Audio endpoint requires nonnegative frame and positive sample rate')
    return round(Fraction(frame * sample_rate, core.FPS))


def _video(path, tail_start, tail_end, *, pixels):
    import av
    frames, count, size = [], 0, None
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('External continuation requires exactly one video stream')
        stream = container.streams.video[0]
        if stream.average_rate != core.FPS:
            raise ValueError('External continuation requires actual 24fps CFR')
        stream.thread_count = 2
        sample_aspect_ratio = stream.sample_aspect_ratio
        # PyAV/FFmpeg reports an unspecified SAR as None or 0/1 depending
        # on the host. Both use the decoded pixel canvas without scaling.
        if sample_aspect_ratio not in (None, Fraction(0, 1), Fraction(1, 1)):
            raise ValueError('External continuation requires square pixels')
        if int(stream.metadata.get('rotate', '0')) % 360:
            raise ValueError('Explicitly convert rotated source as a new material')
        for frame in container.decode(video=0):
            _interrupt()
            if getattr(frame, 'rotation', 0) % 360:
                raise ValueError('Explicitly convert rotated source as a new material')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('External video has no actual frame clock')
            if Fraction(frame.pts) * frame.time_base != Fraction(count, core.FPS):
                raise ValueError('External video must have zero-origin, uninterrupted 24fps frame PTS')
            if size is None:
                size = [frame.width, frame.height]
            if size != [frame.width, frame.height]:
                raise ValueError('External video canvas changes within the source')
            if pixels and tail_start <= count < tail_end:
                frames.append(frame.to_ndarray(format='rgb24'))
            count += 1
    if count < tail_end:
        raise ValueError('Selected external prefix extends beyond actual decoded video')
    return {'fps': _ratio(Fraction(core.FPS, 1)), 'first_frame_pts': _ratio(Fraction(0)),
            'frames': count, 'canvas': size,
            'sample_aspect_ratio': None if sample_aspect_ratio is None else _ratio(sample_aspect_ratio)}, frames


def _pcm_array(frame):
    values = frame.to_ndarray()
    channels = len(frame.layout.channels)
    if not frame.format.is_planar:
        values = values.reshape(-1, channels).T
    if values.shape != (channels, frame.samples):
        raise ValueError('External audio decoder returned an unexpected channel geometry')
    if np.issubdtype(values.dtype, np.integer):
        if values.dtype == np.uint8:
            values = (values.astype(np.float32) - 128) / 128
        else:
            values = values.astype(np.float32) / (2 ** (np.iinfo(values.dtype).bits - 1))
    else:
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('External source audio contains non-finite PCM')
    return values


def _audio(path, tail_start, tail_end, *, pcm, required):
    import av
    parts, rate, channels, first, end, requested = [], None, None, None, None, None
    with av.open(str(path)) as container:
        if len(container.streams.audio) > 1:
            raise ValueError('Choose one explicit original soundtrack before continuation')
        if not container.streams.audio:
            if required:
                raise ValueError('Audio continuation requires an actual original stereo soundtrack')
            return None, None
        for frame in container.decode(audio=0):
            _interrupt()
            if frame.pts is None or frame.time_base is None:
                raise ValueError('External audio has no actual sample clock')
            current_rate, current_channels = frame.sample_rate, len(frame.layout.channels)
            point = Fraction(frame.pts) * frame.time_base * current_rate
            if point.denominator != 1:
                raise ValueError('External audio PTS does not identify an exact source sample')
            point = int(point)
            if rate is None:
                rate, channels, first = current_rate, current_channels, point
                requested = [rounded_audio_frame_sample(tail_start, rate),
                             rounded_audio_frame_sample(tail_end, rate)]
            if current_rate != rate or current_channels != channels or (end is not None and point != end):
                raise ValueError('External audio has changing geometry, gaps or overlapping sample PTS')
            end = point + frame.samples
            if pcm:
                values = _pcm_array(frame)
                left, right = max(point, requested[0]), min(end, requested[1])
                if left < right:
                    parts.append(values[:, left - point:right - point].copy())
    if rate is None:
        raise ValueError('External soundtrack contains no decoded samples')
    if required and (channels != 2 or first > requested[0] or end < requested[1]):
        raise ValueError('Original stereo soundtrack must cover the complete selected motion tail')
    values = np.concatenate(parts, axis=1) if parts else np.zeros((channels, 0), dtype=np.float32)
    if pcm and values.shape != (2, requested[1] - requested[0]):
        raise ValueError('External PCM tail differs from its absolute rounded sample endpoints')
    return {'sample_rate': rate, 'channels': channels, 'first_sample': first,
            'decoded_end_sample': end, 'decoded_samples': end - first,
            'tail_sample_interval': requested,
            'endpoint_policy': 'round_absolute_frame_times_sample_rate_over_24_ties_to_even'}, values


@dataclass(frozen=True)
class ExternalRGBPCMSource:
    root: Path
    contract_json: str

    @property
    def binding(self):
        return json.loads(self.contract_json)

    @property
    def sha256(self):
        return hashlib.sha256(self.contract_json.encode()).hexdigest()

    def verify(self, *, source_binding_json=None):
        binding = self.binding
        if _implementation() != binding['implementation']:
            raise ValueError('External context implementation changed; prepare a new source')
        for key in ('video', 'audio'):
            asset = binding['assets'][key]
            if asset is not None and _file_sha(_inside(self.root, asset['relative_path'])) != asset['sha256']:
                raise ValueError('External source media bytes changed')
        if source_binding_json is not None and _json(json.loads(source_binding_json)) != _json(binding['caller_binding']):
            raise ValueError('External selected version or branch binding changed')
        return binding


def capture_external_rgb_pcm_source(root, relative_path, media_sha256, *, start_frame, end_frame,
                                    context_frames=22, audio_policy='reencode_stereo_pcm_context',
                                    source_binding_json='{}', audio_relative_path=None, audio_sha256=None):
    """Freeze actual source media and a half-open prefix; never register a native parent.

    Optional audio is an explicit separate original soundtrack under the same
    root. Its actual PTS remains authority; reference/refinement audio is not used.
    The caller must independently revalidate its current delivery selection.
    """
    root = Path(root).resolve(strict=True)
    if (type(start_frame) is not int or type(end_frame) is not int
            or not 0 <= start_frame < end_frame or type(context_frames) is not int
            or context_frames not in long_video.CONTEXT_FRAME_STEPS
            or end_frame - start_frame < context_frames or audio_policy not in AUDIO_POLICIES):
        raise ValueError('Select a valid prefix with a complete 5/22/39-frame motion tail')
    caller = json.loads(source_binding_json)
    if not isinstance(caller, dict):
        raise TypeError('External caller binding must be a JSON object')
    _json(caller)
    video = _inside(root, relative_path)
    media_sha256 = _sha(media_sha256)
    if _file_sha(video) != media_sha256:
        raise ValueError('External source video SHA256 does not match')
    if (audio_relative_path is None) != (audio_sha256 is None):
        raise ValueError('Explicit soundtrack path and SHA256 must be supplied together')
    audio = _inside(root, audio_relative_path) if audio_relative_path is not None else video
    expected_audio = _sha(audio_sha256) if audio_sha256 is not None else media_sha256
    if _file_sha(audio) != expected_audio:
        raise ValueError('Original soundtrack SHA256 does not match')
    tail_start = end_frame - context_frames
    clock_video, _ = _video(video, tail_start, end_frame, pixels=False)
    clock_audio, _ = _audio(audio, tail_start, end_frame, pcm=False,
                          required=audio_policy != 'video_only')
    binding = {'schema': SCHEMA, 'origin': 'external_rgb_pcm_reencoded',
        'assets': {'video': {'relative_path': relative_path, 'sha256': media_sha256},
                   'audio': {'relative_path': audio_relative_path or relative_path, 'sha256': expected_audio}},
        'prefix_frame_interval': [start_frame, end_frame], 'tail_frame_interval': [tail_start, end_frame],
        'context_frames': context_frames, 'audio_policy': audio_policy, 'caller_binding': caller,
        'decoded_clock': {'video': clock_video, 'audio': clock_audio}, 'implementation': _implementation(),
        'original_native_latent_available': False, 'has_sampling_ancestor': False,
        'can_resample_original': False, 'automatic_accept': False}
    source = ExternalRGBPCMSource(root, _json(binding))
    source.verify()
    return source


def _producer(component, role):
    import comfy.sd
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    native = {'video_vae': MiniMaxH3VideoVAE, 'audio_vae': MiniMaxH3AudioVAE}
    if type(component) is comfy.sd.VAE and isinstance(component.first_stage_model, native[role]):
        from .progressive_producers import native_producer_identity
        return native_producer_identity(component, role)
    if hasattr(component, 'patcher'):
        identity = nonportable_component_identity(component, 'External VAE wrapper is not portable',
                                                 schema='t8.external-context.nonportable-vae.v1')
    else:
        identity = {'portable_cache_reuse': False, 'component': _execution_selection(component)}
    return {**identity, 'role': role, 'configuration': _execution_selection({
        name: getattr(component, name, None) for name in ('encode', 'audio_sample_rate',
            'latent_channels', 'latent_dim', 'output_channels', 'crop_input')})}


def _inert_tensor(value):
    # Exact tensors can still carry instance method replacements. Reading
    # __dict__ avoids executing an unknown detach/to merely to classify it.
    return type(value) in (torch.Tensor, torch.nn.Parameter) and not vars(value)


def _shared_native_producer(component, role):
    """Only inert native readers may share a snapshot within one guard call.

    This is an optimization predicate, not component admission or a portable
    identity. User wrappers and state getters with callbacks keep their original
    per-context reads. The full producer classifier remains the authority.
    """
    import inspect
    import comfy.model_patcher
    import comfy.sd
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    native = {'video_vae': MiniMaxH3VideoVAE, 'audio_vae': MiniMaxH3AudioVAE}
    if type(component) is not comfy.sd.VAE or type(component.first_stage_model) is not native[role]:
        return False
    patcher = component.patcher
    if (type(patcher) not in (comfy.model_patcher.ModelPatcher, comfy.model_patcher.ModelPatcherDynamic)
            or patcher.model is not component.first_stage_model
            or getattr(patcher.model_state_dict, '__func__', None)
                is not comfy.model_patcher.ModelPatcher.model_state_dict):
        return False
    if any(name in vars(patcher) for name in ('model_state_dict', 'use_ejected',
            'eject_model', 'inject_model', 'get_all_callbacks')):
        return False
    if any(getattr(patcher, name, None) for name in ('patches', 'wrappers', 'callbacks',
            'injections', 'hook_patches', 'forced_hooks', 'current_hooks', 'weight_wrapper_patches')):
        return False
    object_patches = dict(patcher.object_patches)
    cast = object_patches.pop('manual_cast_dtype', None)
    if object_patches or (cast is not None and not isinstance(cast, torch.dtype)):
        return False
    network = component.first_stage_model
    if 'named_modules' in vars(network) or type(network).named_modules is not torch.nn.Module.named_modules:
        return False
    pending, seen = [network], set()
    readers = ('state_dict', '_save_to_state_dict', 'named_modules',
               'named_buffers', 'named_parameters', '_named_members', 'get_extra_state')
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        # Classify getters statically before touching instance attributes.
        # Even reading __dict__/hook collections can execute a custom getter.
        if type(module).__getattribute__ is not object.__getattribute__:
            return False
        if any(inspect.getattr_static(module, name, None) is not getattr(torch.nn.Module, name)
               for name in readers):
            return False
        fields = vars(module)
        if fields.get('_state_dict_pre_hooks') or fields.get('_state_dict_hooks'):
            return False
        for name in ('_parameters', '_buffers'):
            values = fields[name]
            if type(values) is not dict or any(value is not None and not _inert_tensor(value)
                    for value in values.values()):
                return False
        if type(fields['_modules']) is not dict:
            return False
        # Do not execute an unknown child's named_modules getter merely to
        # decide whether its eventual full producer read could be shared.
        pending.extend(child for child in fields['_modules'].values() if child is not None)
    return True


class _ProducerGuard:
    """One call-local full-content snapshot per inert native component and role."""
    def __init__(self):
        self._identities = {}

    def identity(self, component, role):
        key = (id(component), role)
        share = _shared_native_producer(component, role)
        if share and key in self._identities:
            return self._identities[key][1]
        identity = _producer(component, role)
        if share and identity.get('scope') == 'actual_native_weights_tokenizer_configuration_and_implementation':
            # Hold the actual object until this guard ends; no object-id reuse,
            # persistent memo, filename, tensor version or file-stat shortcut.
            self._identities[key] = (component, identity)
        return identity


class ExternalRGBPCMContext(dict):
    def __init__(self, payload, source, video_vae, audio_vae):
        super().__init__(payload)
        self.source, self.video_vae, self.audio_vae = source, video_vae, audio_vae
        self._descriptor_json = _json(self._descriptor())

    def _descriptor(self, producer=None):
        producer = _producer if producer is None else producer
        return {'schema': CONTEXT_SCHEMA, 'origin': self['origin'],
            'native_context_schema': self['schema'], 'empty': self['empty'],
            'source_sha256': self.source.sha256,
            'metadata': self['metadata'], 'video_tail': _tensor_identity(self['video_tail']),
            'audio_tail': _tensor_identity(self['audio_tail']),
            'video_vae': producer(self.video_vae, 'video_vae'),
            'audio_vae': producer(self.audio_vae, 'audio_vae') if self.audio_vae is not None else None,
            'original_native_latent_available': False, 'has_sampling_ancestor': False}


def validate_external_rgb_pcm_context(context, *, source=None, video_vae=None, audio_vae=None):
    """Read-only source/tensor/encoder verification before conditioning or sampling."""
    expected, current = _context_snapshot(context, source=source, video_vae=video_vae, audio_vae=audio_vae)
    if not model_identity_matches(expected, current):
        raise ValueError('External RGB/PCM context tensor, geometry or VAE changed')
    return expected


def _context_snapshot(context, *, source=None, video_vae=None, audio_vae=None, producer=None):
    if type(context) is not ExternalRGBPCMContext:
        raise ValueError('Use an explicit external RGB/PCM context, not an accepted-parent context')
    context.source.verify()
    if source is not None and source is not context.source:
        raise ValueError('External context belongs to another captured source')
    if video_vae is not None and video_vae is not context.video_vae:
        raise ValueError('External context belongs to another video VAE')
    if context.audio_vae is not None and audio_vae is not None and audio_vae is not context.audio_vae:
        raise ValueError('External context belongs to another audio VAE')
    return json.loads(context._descriptor_json), context._descriptor(producer)


def validate_external_rgb_pcm_contexts(contexts, *, source=None, video_vae=None, audio_vae=None):
    """Verify a context group at one boundary with fresh producer descriptions.

    Every context retains its own source, holder, current-object, metadata and
    tensor checks. Only inert native LOW/HIGH producer reads share a snapshot;
    the next call, including a single-context guard, hashes actual weights again.
    No caller-supplied memo or previously computed producer identity is accepted.
    """
    contexts = tuple(contexts)
    if not contexts:
        raise ValueError('External context guard requires an actual context')
    if any(type(context) is not ExternalRGBPCMContext for context in contexts):
        raise ValueError('Use an explicit external RGB/PCM context, not an accepted-parent context')
    if any(not _inert_tensor(context[key]) for context in contexts for key in ('video_tail', 'audio_tail')) or any(
            not _shared_native_producer(context.video_vae, 'video_vae')
            or (context.audio_vae is not None and not _shared_native_producer(context.audio_vae, 'audio_vae'))
            for context in contexts):
        # Unknown tensor readers or producer callbacks retain the original
        # per-context source -> tensor -> producer ordering and full reads.
        return tuple(validate_external_rgb_pcm_context(context, source=source,
            video_vae=video_vae, audio_vae=audio_vae) for context in contexts)
    snapshots = []
    for context in contexts:
        # Check all source/tensor payloads before taking the shared current VAE
        # snapshot, so later context inspection cannot hide an earlier read.
        expected, current = _context_snapshot(context, source=source, video_vae=video_vae,
            audio_vae=audio_vae, producer=lambda component, role: None)
        snapshots.append((context, expected, current))
    guard = _ProducerGuard()
    for context, expected, current in snapshots:
        current['video_vae'] = guard.identity(context.video_vae, 'video_vae')
        current['audio_vae'] = guard.identity(context.audio_vae, 'audio_vae') if context.audio_vae is not None else None
        if not model_identity_matches(expected, current):
            raise ValueError('External RGB/PCM context tensor, geometry or VAE changed')
    return tuple(expected for _, expected, _ in snapshots)


def prepare_external_rgb_pcm_contexts(source, video_vae, audio_vae, *, low_width, low_height,
                                      width, height, context_audio='video_and_audio'):
    """Decode actual selected tail; existing resize/VAEs build both native contexts."""
    if type(source) is not ExternalRGBPCMSource:
        raise ValueError('Capture an explicit external RGB/PCM source before preparing context')
    if context_audio not in {'video_and_audio', 'video_only'}:
        raise ValueError('Select explicit video-only or joint audio continuation')
    binding = source.verify()
    if (context_audio == 'video_and_audio') != (binding['audio_policy'] == 'reencode_stereo_pcm_context'):
        raise ValueError('Context audio policy differs from the captured external source')
    for value in (low_width, low_height, width, height):
        if type(value) is not int or value < 32 or value % 32:
            raise ValueError('External context canvases must be positive multiples of32')
    if low_width >= width or low_height >= height:
        raise ValueError('External LOW canvas must be smaller than HIGH in both dimensions')
    left, right = binding['tail_frame_interval']
    video_path = _inside(source.root, binding['assets']['video']['relative_path'])
    clock_video, pixels = _video(video_path, left, right, pixels=True)
    if clock_video != binding['decoded_clock']['video'] or len(pixels) != binding['context_frames']:
        raise ValueError('External video clock or selected tail changed during decode')
    frames = torch.from_numpy(np.stack(pixels)).float() / 255
    count = binding['context_frames']
    if context_audio == 'video_and_audio':
        if audio_vae is None:
            raise ValueError('Audio continuation requires the connected original audio VAE')
        audio_path = _inside(source.root, binding['assets']['audio']['relative_path'])
        clock_audio, pcm = _audio(audio_path, left, right, pcm=True, required=True)
        if clock_audio != binding['decoded_clock']['audio']:
            raise ValueError('Original soundtrack clock changed during decode')
        with torch.inference_mode():
            audio_tail = core.encode_audio_once(audio_vae,
                {'waveform': torch.from_numpy(pcm).unsqueeze(0), 'sample_rate': clock_audio['sample_rate']})
        bound_audio_vae = audio_vae
    else:
        audio_tail = torch.zeros(1, 32, 2, round(count / core.FPS * core.AUDIO_LATENT_FPS))
        bound_audio_vae = None
    if audio_tail.ndim != 4 or tuple(audio_tail.shape[:3]) != (1, 32, 2):
        raise ValueError('External audio VAE must return native H3 stereo latent geometry')
    if audio_tail.shape[-1] < round(count / core.FPS * core.AUDIO_LATENT_FPS):
        raise ValueError('External audio encode does not cover the complete native context grid')
    delta = int(audio_tail.shape[-1]) - long_video.FRAME_RESCALE * count
    overhang = delta if context_audio == 'video_and_audio' and 0 <= delta < 1 else 0.
    contexts = []
    for stage, w, h in (('low', low_width, low_height), ('high', width, height)):
        _interrupt()
        source.verify()
        with torch.inference_mode():
            tail = video_vae.encode(core.resize_image(frames, w, h)).contiguous()
        if tuple(tail.shape) != (1, 24, long_video.CONTEXT_FRAME_STEPS[count], h // 16, w // 16):
            raise ValueError('External video VAE must return the native H3 context geometry')
        metadata = {'origin': 'external_rgb_pcm_reencoded', 'conditioning_coordinate_only': True,
            'source_segment_index': 0, 'target_segment_index': 1, 'max_context_frames': count,
            'context_audio': context_audio, 'width': w, 'height': h, 'stage': stage,
            'audio_overhang': float(overhang), 'native_audio_latent_end_delta_tokens': float(delta),
            'audio_overhang_policy': 'existing_native_H3_context_grid',
            'audio_context_sample_interval': binding['decoded_clock']['audio']['tail_sample_interval']
                if context_audio == 'video_and_audio' else None,
            'audio_context_source': 'original_PCM_reencoded' if bound_audio_vae is not None else 'unused_placeholder',
            'external_source_sha256': source.sha256, 'original_native_latent_available': False,
            'has_sampling_ancestor': False}
        context = ExternalRGBPCMContext({'schema': long_video.LONG_VIDEO_SCHEMA,
            'origin': 'external_rgb_pcm_reencoded', 'empty': False,
            'video_tail': tail, 'audio_tail': audio_tail, 'metadata': metadata},
            source, video_vae, bound_audio_vae)
        long_video._validate_context(context, 1, count, w, h)
        contexts.append(context)
    source.verify()
    low, high = contexts
    low_identity, high_identity = validate_external_rgb_pcm_contexts(contexts)
    report = {'schema': CONTEXT_SCHEMA, 'source': binding, 'source_sha256': source.sha256,
        'low': low_identity, 'high': high_identity, 'audio_tensor_shared_once_encoded': low['audio_tail'] is high['audio_tail'],
        'additional_sampling_nfe': 0, 'original_native_latent_available': False,
        'has_sampling_ancestor': False, 'delivery_trim_context_frames': count,
        'automatic_accept': False, 'gpu_acceptance': 'NOT_RUN', 'quality_qualified': False}
    return low, high, report


def apply_external_high_prefix(av_latent, high_context, *, source=None, preserve_existing_mask=False):
    """Native full-shape video prefix mask, authenticated by external origin only."""
    from comfy.nested_tensor import NestedTensor

    from .native_masked_context_advanced import (
        _native_audio_mask,
        _native_video_mask,
        require_native_h3_av_mask_support,
    )
    identity = validate_external_rgb_pcm_context(high_context, source=source)
    if high_context['metadata']['stage'] != 'high':
        raise ValueError('Use the actual HIGH external context to constrain its prefix')
    require_native_h3_av_mask_support()
    video, audio = core.nested_av_parts(av_latent)
    if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(audio).all()):
        raise ValueError('External HIGH source AV tensors must be finite')
    tail = high_context['video_tail']
    steps = tail.shape[2]
    if (tuple(video.shape[:2]) != (1, 24) or tuple(video.shape[-2:]) != tuple(tail.shape[-2:])
            or video.shape[2] <= steps):
        raise ValueError('External HIGH prefix must match its canvas and leave new generated frames')
    vm, am = core.split_noise_masks(av_latent, video, audio)
    vm, am = _native_video_mask(vm, video), _native_audio_mask(am, audio)
    if type(preserve_existing_mask) is not bool:
        raise ValueError('Select an explicit external prefix-mask policy')
    if preserve_existing_mask:
        if vm is None or not torch.equal(video[:, :, :steps], tail.to(video)):
            raise ValueError('Preserving an external redraw mask requires its exact reencoded prefix')
        validate_external_rgb_pcm_context(high_context, source=source)
        return av_latent, {'origin': 'external_rgb_pcm_reencoded', 'source_sha256': identity['source_sha256'],
            'context_steps': steps, 'context_frames': high_context['metadata']['max_context_frames'],
            'audio_touched': False, 'original_native_latent_available': False,
            'has_sampling_ancestor': False, 'mode': 'preserved_caller_external_prefix_mask',
            'quality_qualified': False}
    if vm is not None and not bool((vm[:, :, :steps] == 1).all()):
        raise ValueError('External HIGH prefix is already owned by another video mask')
    output_video = video.clone()
    output_video[:, :, :steps] = tail.to(output_video)
    video_mask = torch.ones_like(video) if vm is None else vm.expand_as(video).clone()
    video_mask[:, :, :steps] = 0
    audio_mask = torch.ones_like(audio) if am is None else (am if am.shape == audio.shape else am.expand_as(audio))
    output = {**av_latent, 'samples': NestedTensor((output_video, audio)),
              'noise_mask': NestedTensor((video_mask, audio_mask))}
    validate_external_rgb_pcm_context(high_context, source=source)
    return output, {'origin': 'external_rgb_pcm_reencoded', 'source_sha256': identity['source_sha256'],
        'context_steps': steps, 'context_frames': high_context['metadata']['max_context_frames'],
        'audio_touched': False, 'original_native_latent_available': False,
        'has_sampling_ancestor': False, 'mode': 'native_full_shape_external_video_prefix',
        'quality_qualified': False}
