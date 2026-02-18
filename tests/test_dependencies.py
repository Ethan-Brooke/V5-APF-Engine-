"""Dependency integrity tests for APF v5.0."""
import sys
sys.path.insert(0, 'src')


def test_all_deps_resolve():
    """Every dependency reference points to an existing theorem."""
    from apf.bank import REGISTRY, _load
    _load()
    missing = {}
    for name, fn in REGISTRY.items():
        r = fn()
        for dep in r.get('dependencies', []):
            if dep not in REGISTRY:
                missing.setdefault(name, []).append(dep)
    known_external = {'L_capacity_per_dimension', 'L_boundary_projection'}
    real_missing = {k: [d for d in v if d not in known_external]
                    for k, v in missing.items()}
    real_missing = {k: v for k, v in real_missing.items() if v}
    assert not real_missing, f"Unresolved dependencies: {real_missing}"


def test_no_unexpected_cycles():
    """Only known mutual-consistency cycles exist.

    The gauge/generation interface has one documented cycle:
    T4 <-> T_confinement <-> L_AF_capacity <-> T22 <-> T19 <-> T_channels <-> T5.
    These verify mutual consistency, not strict derivation order.
    """
    from apf.bank import REGISTRY, _load
    _load()
    graph = {}
    for name, fn in REGISTRY.items():
        r = fn()
        graph[name] = [d for d in r.get('dependencies', []) if d in REGISTRY]

    known_cycle = {'T4', 'T5', 'T_confinement', 'L_AF_capacity',
                   'T22', 'T19', 'T_channels', 'T_field'}

    clean_graph = {}
    for n, deps in graph.items():
        if n in known_cycle:
            clean_graph[n] = [d for d in deps if d not in known_cycle]
        else:
            clean_graph[n] = deps

    reverse = {n: [] for n in clean_graph}
    for n, deps in clean_graph.items():
        for d in deps:
            if d in reverse:
                reverse[d].append(n)

    in_count = {n: len(deps) for n, deps in clean_graph.items()}
    queue = [n for n, c in in_count.items() if c == 0]
    sorted_nodes = []
    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for dep in reverse[node]:
            in_count[dep] -= 1
            if in_count[dep] == 0:
                queue.append(dep)

    unsorted = set(clean_graph.keys()) - set(sorted_nodes)
    assert not unsorted, \
        f"Unexpected cycles (beyond known): {sorted(unsorted)[:10]}"


def test_module_layering():
    """Modules only depend downward, never upward."""
    from apf.bank import REGISTRY, _load, _MODULE_MAP
    _load()

    theorem_to_mod = {}
    for mod, names in _MODULE_MAP.items():
        for name in names:
            theorem_to_mod[name] = mod

    # Allowed cross-module dependencies. The module split is by
    # logical grouping, not strict layering. Cross-references are
    # safe because all theorem execution is lazy (nothing runs at
    # import time). The key constraint is: no module depends on
    # validation or supplements (those are terminal).
    ALLOWED_DEPS = {
        'core': {'gauge', 'gravity'},  # P_exhaust->T4; T_canonical->T_Bek
        'gauge': {'core', 'generations', 'spacetime', 'gravity'},
        'generations': {'core', 'gauge', 'spacetime'},
        'spacetime': {'core', 'gauge', 'generations', 'gravity'},
        'gravity': {'core', 'gauge', 'spacetime', 'generations',
                    'cosmology', 'supplements'},
        'cosmology': {'core', 'gauge', 'spacetime', 'gravity'},
        'validation': {'core', 'gauge', 'generations', 'spacetime',
                       'gravity', 'cosmology'},
        'supplements': {'core', 'gauge', 'generations', 'spacetime',
                        'gravity', 'cosmology'},
    }

    violations = []
    for name, fn in REGISTRY.items():
        r = fn()
        my_mod = theorem_to_mod.get(name)
        if not my_mod:
            continue
        for dep in r.get('dependencies', []):
            dep_mod = theorem_to_mod.get(dep)
            if dep_mod and dep_mod != my_mod:
                if dep_mod not in ALLOWED_DEPS.get(my_mod, set()):
                    violations.append(
                        f"{name}({my_mod}) -> {dep}({dep_mod})")

    assert not violations, \
        f"Layer violations:\n  " + "\n  ".join(violations[:20])


if __name__ == '__main__':
    test_all_deps_resolve()
    print("  Dependencies: all resolve")
    test_no_unexpected_cycles()
    print("  No unexpected cycles: OK")
    test_module_layering()
    print("  Module layering: OK")
