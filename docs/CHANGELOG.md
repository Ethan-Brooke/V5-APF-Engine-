# Changelog

## v5.0.0 (2026-02-17)

### Breaking Changes

- **Monolith → 8 modules.** The single-file `FCF_Theorem_Bank_v4_3_6.py`
  has been split into `core`, `gauge`, `generations`, `spacetime`,
  `gravity`, `cosmology`, `validation`, and `supplements`.

- **Import path changed.** `from apf import REGISTRY, run_all` replaces
  direct import of `THEOREM_REGISTRY` from the monolith.

- **Observational constants centralized.** All experimental values
  (Planck 2018, PDG 2024, Fields 2020) moved to `_constants.py`.
  Updating for new data releases requires changing one file.

### New Theorems (21)

Added from v4.3.7 development:

**Gauge (4):**
- `L_anomaly_free` [P] — All 7 SM anomaly cancellation conditions
- `T_proton` [P] — Baryon number exactly conserved, proton stable
- `L_strong_CP_synthesis` [P] — Unified CP: θ=0 (no gain), δ_CKM=π/4 (gain>cost), no axion
- `T_vacuum_stability` [P] — Higgs potential has unique minimum, absolutely stable

**Spacetime (1):**
- `T_Coleman_Mandula` [P] — G = Poincaré × Gauge forced, SUSY excluded

**Gravity (2):**
- `T_graviton` [P] — Massless spin-2, 2 DOF, m=0 from gauge invariance
- `L_Weinberg_Witten` [P] — All particles pass Weinberg-Witten constraints

**Validation (5):**
- `T_concordance` [P/P_structural] — 12 observables, 0 free parameters, mean error 4%
- `T_inflation` [P_structural] — N_e=141, n_s=0.963, r=0.004
- `T_baryogenesis` [P_structural] — η_B = 5.27×10⁻¹⁰
- `T_reheating` [P_structural] — T_rh ~ 10¹⁸ GeV ≫ 1 MeV
- `L_Sakharov` [P] — All 3 Sakharov conditions derived

**Supplements (9):**
- `T_spin_statistics` [P] — Integer↔Bose, half-integer↔Fermi
- `T_CPT` [P] — CPT exact, T violation = CP violation = π/4
- `T_second_law` [P] — dS/dt ≥ 0, arrow of time from L_irr
- `T_decoherence` [P] — Quantum-to-classical transition, no collapse postulate
- `T_Noether` [P] — 26 symmetries ↔ 26 conservation laws
- `T_optical` [P] — S†S = I, optical theorem verified
- `L_cluster` [P] — Exponential correlation decay
- `T_BH_information` [P] — Page curve, information preserved
- `L_naturalness` [P] — Hierarchy dissolved via area-law DOF

### Architecture

- **Lazy registration pattern.** No theorem logic runs at import time.
  Each module exports `register(registry)` which adds function
  references. Actual execution happens in `run_all()`.

- **Two architectural firewalls:**
  - `validation.py` — Observational comparisons isolated from derivations
  - `supplements.py` — Consistency demonstrations isolated from load-bearing proofs

- **Shared infrastructure extracted:**
  - `_result.py` — Uniform result constructor
  - `_linalg.py` — Zero-dependency linear algebra
  - `_helpers.py` — Backward-compatible v4.x wrappers
  - `_constants.py` — Centralized observational data

- **CI pipeline.** GitHub Actions testing on Python 3.10–3.13.

- **Dependency integrity tests.** Automated verification that all
  dependency references resolve, no unexpected cycles exist, and
  module layering is respected.

### Migration Notes

- The `THEOREM_REGISTRY` dict is now `REGISTRY` in `apf.bank`.
- Individual theorem functions are unchanged — they return the same
  result dicts with the same keys.
- The helper functions (`_mm`, `_eigvalsh`, etc.) are now in
  `apf._helpers` but remain backward-compatible.

---

## v4.3.7 (2026-02-17) — Development Only

21 new theorems developed as supplements to v4.3.6. Never released
as a standalone version; merged directly into v5.0.

## v4.3.6 (2026-02-16)

108 theorems in a single monolithic file. The base from which v5.0
was built.
