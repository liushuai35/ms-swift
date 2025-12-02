#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolve layered YAML configs with `imports` and top-level overrides.
- Recursively loads `imports` in order and deep-merges them.
- Then applies the current file's top-level keys (excluding `imports`).
- Output to stdout by default, or `--out` to write a file.

Requirements:
  pip install pyyaml

Usage:
  python ms-swift/tools/resolve_config.py \
    --entry ms-swift/config/experiments/sft_qwen2.5_7b.yaml \
    --out /tmp/final.yaml

Options:
    --json      Output JSON instead of YAML
    --keep-io   Keep `imports` in the output (default drops it)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Union

try:
    import yaml  # type: ignore
except Exception as e:
    print("Missing dependency: pyyaml. Please `pip install pyyaml`.", file=sys.stderr)
    raise

Scalar = Union[str, int, float, bool, None]
Node = Union[Scalar, List[Any], Dict[str, Any]]


def deep_merge(base: Node, override: Node) -> Node:
    """Recursively deep-merge two YAML nodes.
    Rules:
    - If both are dicts: merge keys recursively; override wins on conflicts.
    - Else: override replaces base.
    - Lists: override replaces base (no concat) to keep deterministic behavior.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        out: Dict[str, Any] = dict(base)
        for k, v in override.items():
            if k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    # for lists/scalars or type mismatch: override wins
    return override


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping: {path}")
    return data


def resolve_file(entry_path: str, keep_io: bool = False) -> Dict[str, Any]:
    """Resolve a YAML file with imports and top-level overrides.

    Note: legacy key `overrides` is ignored. Place overriding keys at the top level.
    """
    abs_path = os.path.abspath(entry_path)
    base_dir = os.path.dirname(abs_path)
    node = load_yaml(abs_path)

    imports = node.get('imports') or []
    if imports and not isinstance(imports, list):
        raise ValueError(f"`imports` must be a list in {abs_path}")

    merged: Dict[str, Any] = {}
    for imp in imports:
        if not isinstance(imp, str):
            raise ValueError(f"`imports` entries must be strings: {imp} in {abs_path}")
        imp_path = imp
        if not os.path.isabs(imp_path):
            imp_path = os.path.normpath(os.path.join(base_dir, imp_path))
        part = resolve_file(imp_path, keep_io=keep_io)
        merged = deep_merge(merged, part)

    # remove keys to avoid them being merged prematurely
    node_wo_io = dict(node)
    # Handle legacy `overrides`: ignore but warn to stderr to help migration.
    overrides = node_wo_io.pop('overrides', None)
    if overrides is not None:
        sys.stderr.write(
            f"[resolve_config] WARNING: `overrides` found in {abs_path} but is ignored. "
            "Please move keys to the top level.\n"
        )
    node_wo_io.pop('imports', None)

    # Apply current file's top-level keys after imports merge
    merged = deep_merge(merged, node_wo_io)

    if keep_io:
        # put back imports for debugging if requested
        if imports:
            merged['imports'] = imports
    return merged


def main():
    ap = argparse.ArgumentParser(description="Resolve layered YAML configs with imports and overrides.")
    ap.add_argument('--entry', required=True, help='Entry YAML file (experiment/task/base).')
    ap.add_argument('--out', default=None, help='Output path (YAML). If omitted, print to stdout.')
    ap.add_argument('--json', dest='as_json', action='store_true', help='Output JSON instead of YAML.')
    ap.add_argument('--keep-io', action='store_true', help='Keep imports/overrides in output.')
    args = ap.parse_args()

    result = resolve_file(args.entry, keep_io=args.keep_io)

    if args.as_json:
        text = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        text = yaml.safe_dump(result, allow_unicode=True, sort_keys=False)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
