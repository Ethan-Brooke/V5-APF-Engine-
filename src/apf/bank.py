"""APF v5.0 Theorem Bank — unified registry with lazy loading.

Each physics module exports a register(registry) function that adds
its theorem check functions to the global REGISTRY. No theorem logic
runs at import time; execution happens only when run_all() is called.

Module loading order respects the dependency DAG:
  core → gauge → generations → spacetime → gravity
       → cosmology → validation → supplements
"""

from collections import OrderedDict
from apf._helpers import CheckFailure

__all__ = ['REGISTRY', 'run_all', 'main']

REGISTRY = OrderedDict()

# Module load order (respects dependency DAG)
_MODULE_PATHS = [
    'apf.core',
    'apf.gauge',
    'apf.generations',
    'apf.spacetime',
    'apf.gravity',
    'apf.cosmology',
    'apf.validation',
    'apf.supplements',
]

# Module name -> list of theorem names (populated at load time)
_MODULE_MAP = {}

_loaded = False


def _load():
    """Import all modules and merge their registries."""
    global _loaded
    if _loaded:
        return
    from importlib import import_module
    for mod_path in _MODULE_PATHS:
        mod_name = mod_path.split('.')[-1]
        try:
            mod = import_module(mod_path)
            before = set(REGISTRY.keys())
            mod.register(REGISTRY)
            after = set(REGISTRY.keys())
            _MODULE_MAP[mod_name] = sorted(after - before)
        except ImportError as e:
            import warnings
            warnings.warn(
                f"APF: Failed to load module '{mod_name}': {e}. "
                f"This module will have 0 theorems.",
                RuntimeWarning,
                stacklevel=2,
            )
            _MODULE_MAP[mod_name] = []
    _loaded = True


def run_all(modules=None, verbose=True):
    """Execute all theorem checks.

    Parameters
    ----------
    modules : list[str], optional
        Filter by module name(s). None = run all.
    verbose : bool
        Print results to stdout.

    Returns
    -------
    dict
        {theorem_name: result_dict} for all executed checks.
    """
    _load()

    # Determine which theorems to run
    if modules:
        names_to_run = set()
        for mod in modules:
            names_to_run.update(_MODULE_MAP.get(mod, []))
    else:
        names_to_run = set(REGISTRY.keys())

    results = {}
    passed = failed = errors = 0

    for name, check_fn in REGISTRY.items():
        if name not in names_to_run:
            continue
        try:
            r = check_fn()
            results[name] = r
            if r['passed']:
                passed += 1
                mark = 'PASS'
            else:
                failed += 1
                mark = 'FAIL'
            if verbose:
                ep = r.get('epistemic', '?')
                print(f"  {mark} [{ep:14s}] {name}")
        except CheckFailure as e:
            failed += 1
            results[name] = {'name': name, 'passed': False,
                             'error': str(e), 'epistemic': 'FAIL'}
            if verbose:
                print(f"  FAIL [{'CHECK':14s}] {name}: {e}")
        except Exception as e:
            errors += 1
            results[name] = {'name': name, 'passed': False,
                             'error': str(e), 'epistemic': 'ERROR'}
            if verbose:
                print(f"  ERR  [{'ERROR':14s}] {name}: {e}")

    total = passed + failed + errors
    if verbose:
        print(f"\n  {'='*60}")
        if modules:
            print(f"  Modules: {', '.join(modules)}")
        print(f"  {passed} passed, {failed} failed, "
              f"{errors} errors, {total} total")
        print(f"  {'='*60}")

    return results


def list_modules():
    """Return dict of {module_name: [theorem_names]}."""
    _load()
    return dict(_MODULE_MAP)


def main():
    """CLI entry point."""
    import sys

    args = sys.argv[1:]
    modules = None
    verbose = True

    # Parse args
    i = 0
    while i < len(args):
        if args[i] in ('--module', '-m') and i + 1 < len(args):
            modules = [args[i + 1]]
            i += 2
        elif args[i] == '--list':
            _load()
            for mod, names in _MODULE_MAP.items():
                print(f"\n  {mod} ({len(names)} theorems)")
                for n in names:
                    print(f"    {n}")
            total = sum(len(v) for v in _MODULE_MAP.values())
            print(f"\n  Total: {total} theorems in "
                  f"{len(_MODULE_MAP)} modules")
            sys.exit(0)
        elif args[i] == '--quiet':
            verbose = False
            i += 1
        elif args[i] in ('--help', '-h'):
            print("Usage: python -m apf.bank [OPTIONS]")
            print("  --module NAME    Run only one module")
            print("  --list           List all modules and theorems")
            print("  --quiet          Suppress output")
            print("  --help           Show this help")
            sys.exit(0)
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)

    print(f"\n  APF v5.0 Theorem Bank")
    print(f"  {'='*60}\n")
    results = run_all(modules=modules, verbose=verbose)
    sys.exit(0 if all(r.get('passed', False)
                      for r in results.values()) else 1)


if __name__ == '__main__':
    main()
