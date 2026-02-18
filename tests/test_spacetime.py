"""Tests for apf.spacetime module."""
import sys
sys.path.insert(0, 'src')


def test_spacetime_all_pass():
    from apf.bank import run_all
    results = run_all(modules=['spacetime'], verbose=False)
    failures = [n for n, r in results.items() if not r.get('passed')]
    assert not failures, f"Failed: {failures}"


def test_spacetime_count():
    from apf.bank import list_modules
    mods = list_modules()
    assert len(mods['spacetime']) > 0, "Module has no theorems"


def test_spacetime_result_format():
    """Every theorem returns a well-formed result dict."""
    from apf.bank import REGISTRY, _load
    _load()
    from apf.bank import _MODULE_MAP
    required_keys = {'name', 'tier', 'passed', 'epistemic',
                     'summary', 'key_result', 'dependencies'}
    for name in _MODULE_MAP.get('spacetime', []):
        r = REGISTRY[name]()
        missing = required_keys - set(r.keys())
        assert not missing, f"{name} missing keys: {missing}"
        assert isinstance(r['tier'], (int, float)), f"{name}: tier not numeric"
        assert isinstance(r['passed'], bool), f"{name}: passed not bool"
        assert isinstance(r['dependencies'], list), f"{name}: deps not list"


if __name__ == '__main__':
    test_spacetime_all_pass()
    test_spacetime_count()
    test_spacetime_result_format()
    print(f"  spacetime: all tests passed")
