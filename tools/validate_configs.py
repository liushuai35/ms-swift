#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate all YAML files under `ms-swift/config` for syntax correctness.

Requirements:
  pip install pyyaml

Usage:
  python ms-swift/tools/validate_configs.py
"""
from __future__ import annotations
import os
import sys
from typing import List

try:
    import yaml  # type: ignore
except Exception as e:
    print("Missing dependency: pyyaml. Please `pip install pyyaml`.", file=sys.stderr)
    raise

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))


def find_yaml_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(('.yml', '.yaml')):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def main():
    files = find_yaml_files(ROOT)
    ok = True
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as h:
                yaml.safe_load(h)
        except Exception as e:
            ok = False
            print(f"[ERROR] YAML invalid: {f}: {e}")
        else:
            print(f"[OK]     {f}")
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
