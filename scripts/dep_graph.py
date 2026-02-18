#!/usr/bin/env python3
"""Generate dependency graph as DOT or text summary."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from apf.bank import REGISTRY, _load, _MODULE_MAP
_load()

theorem_to_mod = {}
for mod, names in _MODULE_MAP.items():
    for name in names:
        theorem_to_mod[name] = mod

print("Module-level dependency summary:\n")
for mod in ['core','gauge','generations','spacetime','gravity','cosmology','validation','supplements']:
    deps_on = set()
    for name in _MODULE_MAP.get(mod, []):
        r = REGISTRY[name]()
        for dep in r.get('dependencies', []):
            dep_mod = theorem_to_mod.get(dep)
            if dep_mod and dep_mod != mod:
                deps_on.add(dep_mod)
    if deps_on:
        print(f"  {mod} depends on: {', '.join(sorted(deps_on))}")
    else:
        print(f"  {mod} (no external deps)")

print(f"\nTotal: {len(REGISTRY)} theorems in {len(_MODULE_MAP)} modules")
