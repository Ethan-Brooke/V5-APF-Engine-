# Admissibility Physics Framework (APF) v5.0

An executable theorem bank deriving the Standard Model of particle physics
and cosmological observables from a single axiom: **finite enforceability
of distinguishable records** (A1).

## What This Is

129 machine-verifiable theorems organized into 8 modules, covering:

- **Quantum mechanics** (Born rule, unitarity, decoherence)
- **Gauge theory** (SU(3)×SU(2)×U(1), Higgs mechanism, confinement)
- **Particle content** (3 generations, 61 species, mass hierarchies)
- **Spacetime** (d=4, Lorentzian signature, continuum emergence)
- **Gravity** (Einstein equations, Bekenstein bound, graviton)
- **Cosmology** (density fractions, dark matter, baryon asymmetry)

Every theorem is a Python function that either passes or fails. No external
dependencies — the entire bank runs on the Python standard library.

## Quick Start

```bash
git clone https://github.com/<your-org>/apf.git
cd apf

# Run all 129 theorems
PYTHONPATH=src python -m apf.bank

# Run a single module
PYTHONPATH=src python -m apf.bank --module gravity

# List all modules and theorems
PYTHONPATH=src python -m apf.bank --list

# Run tests
PYTHONPATH=src python -m pytest tests/ -v
```

Or install as a package:

```bash
pip install -e .
python -m apf.bank
```

## Architecture

```
src/apf/
├── core.py           27 theorems   Axioms + quantum bedrock
├── gauge.py          22 theorems   Gauge origin → fields → Higgs → CP
├── generations.py    46 theorems   Mass/mixing monolith
├── spacetime.py       8 theorems   Arena emergence + architecture constraints
├── gravity.py         8 theorems   Dynamics on the arena (kept tight)
├── cosmology.py       4 theorems   Derived cosmological counting
├── validation.py      5 theorems   Observational comparisons (firewall)
├── supplements.py     9 theorems   Consistency exhibitions (firewall)
└── bank.py                         Lazy registry + CLI
```

Two architectural firewalls protect the derivation chain:

- **Validation firewall**: Theorems comparing predictions to observed data
  (Planck, PDG, BBN) are isolated from the derivation itself. Updating
  error bars or fitting formulae cannot contaminate proofs.

- **Supplements firewall**: Theorems that demonstrate consistency
  (decoherence, spin-statistics, naturalness) are separated from
  load-bearing derivations. A reviewer who skips supplements still sees
  the complete A1 → SM → cosmology pipeline.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full module map
and dependency DAG.

## Epistemic Tags

Every theorem carries a tag classifying its proof status:

| Tag | Meaning |
|-----|---------|
| **P** | Proved from the axiom chain (A1 + lemmas) |
| **P_structural** | Structural argument; qualitative conclusion is [P] but specific numbers are model-dependent |
| **P_imported** | Uses an external theorem (Gleason, Lovelock, etc.) whose hypotheses are verified from [P] |
| **W** | Numerical witness, not a proof |
| **O** | Open — claimed but not yet proved |

See [docs/EPISTEMIC_TAGS.md](docs/EPISTEMIC_TAGS.md) for definitions and examples.

## Key Results

| Observable | Framework | Observed | Error |
|-----------|-----------|----------|-------|
| Ω_Λ | 0.6885 | 0.6889 ± 0.0056 | 0.06% |
| Ω_DM | 0.2623 | 0.2607 ± 0.0050 | 0.6% |
| Ω_b | 0.0492 | 0.0490 ± 0.0003 | 0.4% |
| sin²θ_W | 0.2222 | 0.2312 ± 0.0001 | 3.9% |
| n_s | 0.963 | 0.9649 ± 0.0042 | 0.2% |
| Y_p | 0.2467 | 0.2449 ± 0.0040 | 0.7% |
| N_eff | 3.044 | 2.99 ± 0.17 | 0.3σ |

All from **zero free parameters** — the framework derives these from
counting and geometry, not fitting.

## Requirements

- Python ≥ 3.10
- No external dependencies (stdlib only)
- pytest for running tests (optional)

## Repository Contents

```
apf/
├── src/apf/          Python package (8 physics modules + infrastructure)
├── tests/            Per-module tests + integration + dependency integrity
├── docs/             Architecture, epistemic tags, changelog
├── engine/           Admissibility physics engine (artifact)
└── .github/          CI pipeline (GitHub Actions)
```

## Engine

The `engine/` directory contains the Admissibility Physics Engine, a
standalone artifact that implements the enforcement potential, capacity
calculations, and particle spectrum derivation. It is included for
reference alongside the theorem bank but is not part of the bank's
verification pipeline.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Papers documenting the framework are being updated to reference v5.0.
Cross-references between theorems and paper sections will be added in v5.1.
