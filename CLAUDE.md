# CLAUDE.md — APF v5.0 (Admissibility Physics Framework)

## What This Is

An executable theorem bank that derives the Standard Model and cosmological observables from a single axiom (A1: finite enforcement capacity). 129 machine-verifiable theorems across 8 physics modules, zero external dependencies.

## Quick Commands

```bash
# Run all 129 theorems
PYTHONPATH=src python -m apf.bank

# Run a single module
PYTHONPATH=src python -m apf.bank --module core

# List all theorems
PYTHONPATH=src python -m apf.bank --list

# Run tests
PYTHONPATH=src python -m pytest tests/ -v

# Run a specific test file
PYTHONPATH=src python -m pytest tests/test_core.py -v
```

## Project Layout

```
src/apf/
  __init__.py        # Package entry, exports REGISTRY and run_all
  bank.py            # Lazy registry, CLI, execution engine
  _result.py         # Result constructor & epistemic tag definitions
  _constants.py      # Observational data (Planck 2018, PDG 2024, BBN)
  _helpers.py        # v4.x compatibility wrappers
  _linalg.py         # Stdlib-only linear algebra (complex matrices)
  core.py            # 27 theorems — axiom A1, quantum skeleton
  gauge.py           # 22 theorems — SU(3)×SU(2)×U(1), Higgs
  generations.py     # 46 theorems — mass/mixing hierarchies, CKM, PMNS
  spacetime.py       # 8 theorems  — 4D Lorentzian arena
  gravity.py         # 8 theorems  — Einstein equations, Bekenstein bound
  cosmology.py       # 4 theorems  — density fractions, dark matter
  validation.py      # 5 theorems  — observational comparisons (firewall)
  supplements.py     # 9 theorems  — consistency demonstrations (firewall)
tests/
  test_bank.py       # Integration: all 129 pass, module counts, no dupes
  test_core.py .. test_supplements.py  # Per-module validation
  test_dependencies.py  # DAG integrity, cycle detection, layering
  test_mutations.py     # Mutation testing
docs/
  ARCHITECTURE.md    # Module DAG, design principles, tier system
  EPISTEMIC_TAGS.md  # Proof status tags (P, P_structural, W, etc.)
  CHANGELOG.md       # Release notes, migration from v4.x
engine/              # Standalone monolithic computation artifact
scripts/             # Utilities (dep_graph.py, run_all.py)
```

## Module Dependency DAG

```
A1 (axiom)
  ↓
core (27) ──→ gauge (22) ──→ generations (46)
  │
  ├──→ spacetime (8)
  └──→ gravity (8) ──→ cosmology (4)

Terminal (firewall):
  validation (5)   — compares predictions to observed data
  supplements (9)  — consistency checks
```

Firewalled modules are terminal — nothing depends on them. This prevents observational data from contaminating derivations.

## Architecture Rules

- **Zero external dependencies.** Stdlib only (math, cmath, fractions). No numpy, scipy, or any third-party package.
- **Lazy registration.** Each module exports `register(registry)`. No theorem logic runs at import time.
- **Standardized results.** Every theorem function returns a dict via `_result()` with: `name`, `tier`, `passed`, `epistemic`, `summary`, `key_result`, `dependencies`, `artifacts`.
- **Module layering.** The dependency DAG must remain acyclic (tested in `test_dependencies.py`). Validation and supplements are terminal firewalls.

## Coding Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Theorem functions | `check_<NAME>()` | `check_A1()`, `check_T_Born()` |
| Internal modules | Underscore prefix | `_linalg.py`, `_constants.py` |
| Constants | SCREAMING_SNAKE_CASE | `REGISTRY`, `PLANCK`, `PHYSICAL` |
| Internal helpers | Underscore prefix, snake_case | `_mm()`, `_dag()` |
| Classes | PascalCase | `CheckFailure`, `Epistemic` |
| Docstrings | STATEMENT / CONTENT / STATUS sections | See any `check_*` function |

## Epistemic Tags

Every theorem declares its proof status:

- **P** — Proved from the axiom chain
- **P_structural** — Structural argument; qualitative proof, specific numbers model-dependent
- **P_imported** — Uses an external theorem (Gleason, Lovelock, etc.)
- **AXIOM** — Foundational (A1 only)
- **POSTULATE** — Not derived from A1 (M, NT)
- **W** — Numerical witness, not a proof
- **O** — Open (claimed but incomplete)

## Adding a New Theorem

1. Write `check_<NAME>()` in the appropriate module, returning `_result(...)`.
2. Register it in that module's `register()` function.
3. Declare `dependencies` accurately — the DAG tests enforce this.
4. Assign the correct `tier` and `epistemic` tag.
5. Add a test in the corresponding `tests/test_<module>.py`.
6. Run `PYTHONPATH=src python -m pytest tests/ -v` — all 129+ theorems must pass.

## Common Pitfalls

- **Don't import third-party packages.** The stdlib-only constraint is load-bearing.
- **Don't create circular dependencies** between physics modules. The DAG is tested.
- **Don't put derivation logic in validation.py or supplements.py.** They are firewall modules — terminal only.
- **Always set PYTHONPATH=src** when running from the repo root (unless installed via `pip install -e .`).
