"""Official installation audit and guarded regular Topaz worker orchestration."""
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .topaz_contract import OfficialTopaz, REGULAR_PARAMETERS, regular_filter
from .topaz_media import file_identity


def _run_readonly(command, runtime, *, timeout=30, environment=None):
    result = subprocess.run(command, cwd=runtime.install,
        env=environment or runtime.child_environment(os.environ), capture_output=True,
        timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), shell=False)
    if result.returncode:
        raise RuntimeError('Official runtime inspection failed, exit code ' + str(result.returncode)
            + ': ' + result.stderr.decode('utf8', 'replace')[-2400:])
    return (result.stdout + result.stderr).decode('utf8', 'replace')


def validate_signatures(signatures, paths):
    if (not isinstance(signatures, list) or len(signatures) != len(paths)
            or any(not isinstance(s, dict) or s.get('path') != str(p)
                or s.get('status') != 'Valid'
                or 'O=Topaz Labs LLC' not in (s.get('signer') or '')
                for s, p in zip(signatures, paths))):
        raise RuntimeError('All selected executables must have valid official Topaz Labs signatures')


def audit_installation(runtime: OfficialTopaz):
    if os.name != 'nt':
        raise RuntimeError('This official Topaz integration currently requires Windows')
    files = [runtime.executable(name) for name in ('Topaz Video.exe', 'ffmpeg.exe', 'ffprobe.exe')]
    identities = [file_identity(path) for path in files]
    # Fixed PowerShell code; paths travel as JSON in a child-only variable, never
    # interpolated into shell text. Do not inspect license/login files.
    code = ("$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8; "
        "Import-Module Microsoft.PowerShell.Security -ErrorAction Stop; "
        "$paths=ConvertFrom-Json $env:T8_TOPAZ_AUDIT_PATHS; $rows=foreach($path in $paths) { "
        "$s=Get-AuthenticodeSignature -LiteralPath $path; "
        "[pscustomobject]@{path=$path;status=[string]$s.Status;signer=$s.SignerCertificate.Subject;"
        "version=(Get-Item -LiteralPath $path).VersionInfo.FileVersion} }; ConvertTo-Json -InputObject @($rows) -Compress")
    ps = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    env = runtime.child_environment(os.environ)
    # Do not inherit PowerShell7 module paths into Windows PowerShell5.1.
    # Its built-in security module performs signature checks; no policy bypass.
    env = {k: v for k, v in env.items() if k.upper() != 'PSMODULEPATH'}
    env['T8_TOPAZ_AUDIT_PATHS'] = json.dumps([str(p) for p in files])
    signatures = json.loads(_run_readonly([str(ps), '-NoProfile', '-NonInteractive', '-EncodedCommand',
        base64.b64encode(code.encode('utf-16-le')).decode('ascii')], runtime, environment=env).lstrip('\ufeff'))
    validate_signatures(signatures, files)
    if [file_identity(path) for path in files] != identities:
        raise RuntimeError('Installation changed during audit')
    help_text = _run_readonly([str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-h', 'filter=tvai_up'], runtime)
    options = sorted(set(re.findall(r'^\s+(\w+)\s+<', help_text, re.M)))
    required = {'model', 'scale', 'w', 'h', 'device', 'instances', 'download', 'vram'}
    if 'Filter tvai_up' not in help_text or not required.issubset(options):
        raise RuntimeError('This official FFmpeg lacks the required tvai_up interface')
    return {'status': 'official_interface_verified_not_model_inference',
        'executables': identities, 'signatures': signatures, 'tvai_up_options': options,
        'neuroserver_directory_present': (runtime.install / 'neuroserver').is_dir(),
        'model_catalog': model_catalog(runtime, options),
        'downloads': False, 'license_or_login_read': False}


def _candidate_weights(runtime, definition, scale, files=None):
    short, version = definition.get('shortName'), definition.get('version')
    if (not isinstance(short, str) or not re.fullmatch('[a-z0-9-]+', short)
            or type(version) not in (str, int) or not re.fullmatch('[0-9]{1,6}', str(version))):
        raise ValueError('Unsupported model definition naming contract')
    prefix = f'{short}-v{version}-'
    files = runtime.data.iterdir() if files is None else files
    weights = [p for p in files if p.is_file() and p.name.startswith(prefix)
        and p.suffix in ('.tz', '.tz3') and f'-{scale}x-' in p.name]
    if any(p.resolve(strict=True).parent != runtime.data for p in weights):
        raise ValueError('Candidate model weights leave the selected data directory')
    return sorted(weights)


def model_catalog(runtime, runtime_options):
    """Read-only directory inventory, never a license or execution readiness probe."""
    models, skipped = [], []
    files = list(runtime.data.iterdir())
    options = set(runtime_options)
    for entry in sorted(runtime.definitions.glob('*.json')):
        try:
            path, definition = runtime.model(entry.stem)
            row = {'id': entry.stem, 'definition_path': str(path),
                   'execution_verified': False, 'enabled_in_definition': definition.get('enabled'),
                   'definition_name': definition.get('displayName') or definition.get('name') or definition.get('shortName') or entry.stem}
            if definition.get('isNeuroserverModel'):
                row.update(route='neuroserver', status='separate_neuroserver_qualification_required')
            elif definition.get('changesFPS') or definition.get('modelType') != 1:
                row.update(route='not_offered_by_regular_upscale',
                           status='fps_auxiliary_or_unclassified_definition',
                           declared_model_type=definition.get('modelType'))
            else:
                row.update(route='tvai_up', status='definition_discovered', scales={})
                for scale in (1, 2, 4):
                    weights = _candidate_weights(runtime, definition, scale, files)
                    row['scales'][str(scale)] = {
                        'status': 'candidate_files_present_unverified' if weights else 'missing_candidate_weights',
                        'candidate_files': [{'name': p.name, 'bytes': p.stat().st_size} for p in weights]}
                parameters = {}
                for item in definition.get('parameters', []):
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get('name', '')).lower()
                    if name in REGULAR_PARAMETERS & options and name not in ('estimate', 'kcolor', 'blend'):
                        parameters[name] = {key: item[key] for key in
                            ('min', 'max', 'default', 'guiName') if key in item}
                        parameters[name]['source'] = 'selected_model_definition'
                for name, maximum in (('blend', 1), ('estimate', 100), ('kcolor', 1)):
                    if name in options:
                        parameters[name] = {'min': 0, 'max': maximum, 'source': 'runtime_option',
                                            'integer_only': name != 'blend'}
                row['parameters'] = parameters
            models.append(row)
        except (OSError, ValueError, TypeError) as error:
            skipped.append({'id': entry.stem, 'reason': str(error)})
    return {'models': models, 'unreadable_or_nonmodel_definitions': skipped,
            'inference_executed': False, 'weight_hashes_computed': False,
            'scope': 'Definitions and scale-specific candidate filenames only; engine loading, license and output are not verified. Exact weight identities are checked by enhancement execution.'}


def model_evidence(runtime, model_id, scale):
    path, definition = runtime.model(model_id)
    if definition.get('isNeuroserverModel'):
        raise ValueError('Formal Starlight requires a separately verified Neuroserver route')
    weights = _candidate_weights(runtime, definition, scale)
    if not weights:
        raise RuntimeError(f'{model_id}: no installed {scale}x weight candidates. Download the selected model through official Topaz first; automatic downloading is disabled.')
    return {'definition': file_identity(path), 'scale': scale,
        'candidate_weights': [file_identity(p) for p in sorted(weights)],
        'status': 'candidate_files_present_actual_engine_load_still_required'}


def dimension_model_evidence(runtime, model_id):
    """Bind available variants; engine-selected variant is not assumed from size."""
    path, definition = runtime.model(model_id)
    if definition.get('isNeuroserverModel') or definition.get('changesFPS') or definition.get('modelType', 1) != 1:
        raise ValueError('Custom dimensions require a regular enhancement model')
    files = list(runtime.data.iterdir())
    by_scale = {scale: _candidate_weights(runtime, definition, scale, files) for scale in (1, 2, 4)}
    weights = sorted({p for candidates in by_scale.values() for p in candidates})
    if not weights:
        raise RuntimeError(f'{model_id}: no installed weight candidates; prepare the model through official Topaz first')
    return {'definition': file_identity(path), 'scale': 'engine_auto_dimensions',
        'candidate_weights': [file_identity(p) for p in weights],
        'missing_candidate_scales': [scale for scale, candidates in by_scale.items() if not candidates],
        'actual_engine_variant_verified': False,
        'status': 'candidate_files_present_actual_engine_load_still_required'}


def validate_publication(job, spec, report):
    if report.get('status') != 'media_audit_pass_human_pending':
        raise RuntimeError('Worker did not produce a verified enhancement publication')
    if report.get('source') != spec['source'] or file_identity(spec['source']['path']) != spec['source']:
        raise RuntimeError('Source identity changed before publication')
    expected = report.get('output', {})
    candidate = Path(expected.get('path', '')).resolve(strict=True)
    if candidate.parent != Path(job).resolve(strict=True) or candidate.name not in ('enhanced.mov', 'enhanced.mkv'):
        raise RuntimeError('Output publication must be the completed task-owned master')
    if file_identity(candidate) != expected:
        raise RuntimeError('Output identity changed before publication')
    return candidate


def run_regular(runtime, source, job, *, model_id, width, height, scale,
                lease_path, device=0, vram=.8, parameters=None, interrupt=None, size_mode='scale'):
    """Single owned task, no auto retry/resume, only publish after media audit.

    Fixed scales and explicit same-aspect sizes are checked against actual source
    geometry by the owned worker before enhancement. No parent video decode.
    """
    from .dlss_fi_backend.process import run_isolated, IsolatedTaskError
    from .dlss_fi_backend.resources import SerialProbeLease, NvmlResourceReader, ResourceGuard
    if scale not in (1, 2, 4) or type(scale) is not int or device != 0:
        raise ValueError('Initial regular Topaz route supports fixed1x/2x/4x on a single GPU0')
    if (width is None) != (height is None):
        raise ValueError('Either provide both target dimensions or infer both from the source')
    if size_mode not in ('scale', 'target_dimensions') or (size_mode == 'target_dimensions' and width is None):
        raise ValueError('Custom size mode requires both target dimensions')
    regular_filter(runtime, model_id, width if width is not None else 32,
        height if height is not None else 32, device=device, vram=vram, parameters=parameters)
    source = Path(source).resolve(strict=True)
    job = Path(job).resolve()
    if job.exists():
        raise ValueError('Use a new task directory; source and previous outputs are never overwritten')
    installation = audit_installation(runtime)
    evidence = (dimension_model_evidence(runtime, model_id) if size_mode == 'target_dimensions'
        else model_evidence(runtime, model_id, scale))
    if parameters and not set(parameters).issubset(installation['tvai_up_options']):
        raise ValueError('Selected parameter is not supported by the installed FFmpeg')
    # Conservative upper bound for an uncompressed lossless master is refined
    # after full source probe in the owned worker, before enhancement starts.
    job.mkdir(parents=True)
    spec = {'install': str(runtime.install), 'definitions': str(runtime.definitions), 'data': str(runtime.data),
        'source': file_identity(source), 'installation': installation, 'model': evidence,
        'settings': {'model_id': model_id, 'width': width, 'height': height, 'scale': scale, 'size_mode': size_mode,
            'device': device, 'vram': vram, 'parameters': parameters or {}}, 'job': str(job)}
    (job / 'request.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
    receipt = {'status': 'incomplete', 'no_automatic_downloads': True}
    interrupt_error = None
    try:
        with SerialProbeLease(lease_path), NvmlResourceReader() as reader:
            guard = ResourceGuard()
            reason = guard.observe(reader.sample(), startup=True)
            if reason:
                raise RuntimeError('Topaz startup resource guard: ' + reason)
            previous = time.monotonic()
            with (job / 'resources.jsonl').open('x', encoding='utf8') as log:
                def check():
                    nonlocal previous, interrupt_error
                    if interrupt:
                        try:
                            interrupt()
                        except BaseException as error:
                            interrupt_error = error
                            raise
                    if time.monotonic() - previous < .25:
                        return
                    previous = time.monotonic()
                    row = reader.sample()
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    reason = guard.observe(row)
                    if reason:
                        raise RuntimeError('Topaz resource guard: ' + reason)
                    if shutil.disk_usage(job).free < 1024**3:
                        raise RuntimeError('Topaz stopped before filling output disk')
                receipt = run_isolated(Path(__file__).with_name('topaz_worker.py'),
                    [str(job / 'request.json')], timeout=3600, check=check)
    except IsolatedTaskError as error:
        receipt = error.receipt
        # The generic owned-process runner deliberately captures exceptions to
        # ensure job cleanup. Restore Comfy's actual cancellation exception only
        # once it certifies no owned descendants remain; never mask cleanup failure.
        if interrupt_error is not None and receipt.get('active_after_cleanup') == 0:
            raise interrupt_error from error
        raise
    except BaseException as error:
        receipt = {'status': 'controller_failed', 'error': str(error)}
        raise
    finally:
        (job / 'process.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    report = json.loads((job / 'result.json').read_text(encoding='utf8'))
    return validate_publication(job, spec, report), report
