"""Integration tests for the full APF v5.0 theorem bank."""
import sys
sys.path.insert(0, 'src')


def test_full_bank_129():
    """All 129 theorems pass."""
    from apf.bank import run_all
    results = run_all(verbose=False)
    failures = [n for n, r in results.items() if not r.get('passed')]
    assert len(results) == 129, f"Expected 129, got {len(results)}"
    assert not failures, f"Failed: {failures}"


def test_module_counts():
    """Each module has the expected number of theorems."""
    from apf.bank import list_modules
    expected = {
        'core': 27,
        'gauge': 22,
        'generations': 46,
        'spacetime': 8,
        'gravity': 8,
        'cosmology': 4,
        'validation': 5,
        'supplements': 9,
    }
    mods = list_modules()
    for mod, count in expected.items():
        actual = len(mods.get(mod, []))
        assert actual == count, f"{mod}: expected {count}, got {actual}"


def test_no_duplicate_names():
    """No two theorems share a name."""
    from apf.bank import REGISTRY, _load
    _load()
    names = list(REGISTRY.keys())
    assert len(names) == len(set(names)), \
        f"Duplicates: {[n for n in names if names.count(n) > 1]}"


def test_epistemic_tags_valid():
    """All epistemic tags are from the allowed set."""
    from apf.bank import REGISTRY, _load
    _load()
    valid = {'AXIOM', 'POSTULATE', 'P', 'P_structural', 'P_imported',
             'W', 'O', 'P/P_structural'}
    for name, fn in REGISTRY.items():
        r = fn()
        assert r['epistemic'] in valid, \
            f"{name}: unknown epistemic tag '{r['epistemic']}'"


if __name__ == '__main__':
    test_full_bank_129()
    print("  Full bank: 129/129 PASS")
    test_module_counts()
    print("  Module counts: OK")
    test_no_duplicate_names()
    print("  No duplicates: OK")
    test_epistemic_tags_valid()
    print("  Epistemic tags: OK")
