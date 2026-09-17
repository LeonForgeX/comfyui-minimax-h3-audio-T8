"""Inventory all owned runtime sources and compatibility guard candidates; no inference.

Static candidates are review evidence, not a proof that every composition works.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


PATTERN = re.compile(r'lora|sage|sol.att|owner|object.patch|callback|wrapper|injection|hook|'
                     r'unverified|unqualified|unaudited|authentic|compatib|conflict|refus|reject', re.I)
EXCLUDED = {'vendor', 'flashvsr_vendor', 'raven_vendor', '__pycache__'}
SOL_SHA256 = '931c3602d7433a1dad313aebbbbe13067fdf6c1c607cdcb1b52ac173ae21e5e6'


def inventory(project):
    paths = list(project.glob('*.py')) + [p for p in (project / 'h3_t8').rglob('*.py')
        if not EXCLUDED.intersection(p.relative_to(project).parts)]
    sources, guards, advisories = {}, [], []
    for path in sorted(paths):
        payload = path.read_bytes()
        tree = ast.parse(payload.decode('utf-8-sig'), filename=str(path))
        relative = path.relative_to(project).as_posix()
        sources[relative] = hashlib.sha256(payload).hexdigest()
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            kind = None
            if isinstance(node, ast.Raise) and node.exc is not None:
                kind = 'raise_review_candidate'
            elif isinstance(node, ast.Return):
                kind = 'return_review_candidate'
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'warn_patch_stack':
                kind = 'compatibility_advisory'
            if kind is None:
                continue
            literals = ' '.join(item.value for item in ast.walk(node)
                                if isinstance(item, ast.Constant) and isinstance(item.value, str))
            if kind != 'compatibility_advisory' and not PATTERN.search(literals):
                continue
            owner = node
            while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = parents[owner]
            item = dict(file=relative, line=node.lineno, kind=kind,
                        function=getattr(owner, 'name', '<module>'),
                        literals=literals, source=ast.unparse(node))
            (advisories if kind == 'compatibility_advisory' else guards).append(item)
    if sources.get('sol_attn_minimax_v2.py') != SOL_SHA256:
        raise RuntimeError('Original registration-only Sol source changed')
    return dict(schema='t8.h3.patch_stack_guard_inventory/v1', sources=sources,
                runtime_source_count=len(sources), syntax_valid=True,
                guard_candidates=guards, advisories=advisories,
                excluded_vendor_directories=sorted(EXCLUDED),
                immutable_registration_only_sources=['sol_attn_minimax_v2.py'],
                boundaries=['AST inventory is not behavioral compatibility proof.',
                            'UnverifiedModelStack is a portable-cache classifier; callers must fall back.',
                            'Optional inspection is advisory; own receipt/hash validators and real errors remain.',
                            'No GPU generation, queue, network, Git mutation or publication.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if not destination.is_relative_to(project / 'artifacts') or destination.exists():
        raise ValueError('Choose a new artifact JSON path inside the project')
    report = inventory(project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('runtime_source_count', 'syntax_valid')}))
    print('guard_candidates', len(report['guard_candidates']), 'advisories', len(report['advisories']))


if __name__ == '__main__':
    main()
