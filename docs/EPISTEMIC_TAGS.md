# Epistemic Tags in APF v5.0

Every theorem in the bank carries an epistemic tag classifying the
strength of its derivation. This is not decoration — it is the
framework's mechanism for intellectual honesty about what has actually
been proved vs. what has been checked vs. what remains open.

## Tag Definitions

### P — Proved

The theorem follows from the axiom chain (A1 + previously proved
lemmas and theorems) via a logical derivation that can be
independently verified. The proof may use standard mathematical
techniques (linear algebra, calculus, combinatorics) but does not
import any external physics.

**Example:** T_Born — the Born rule follows from A1 + T2 + Gleason's
theorem. Gleason's theorem is a mathematical result whose hypotheses
(dimension ≥ 3, frame function on closed subspaces) are verified
from A1.

### P_structural — Structural Argument

The qualitative conclusion is [P] — it follows from the axiom chain.
But the specific numerical value involves modelling choices (e.g.,
slow-roll approximation, perturbative decay rates, leading-order
estimates) that are standard physics but not uniquely forced.

**Example:** T_inflation — the framework derives that inflation occurs
with N_e ≫ 60 e-folds. The specific value N_e = 141 uses the
slow-roll approximation and a particular model for the enforcement
potential's inflationary plateau. The qualitative statement
"sufficient inflation occurs" is [P]; the number 141 is
[P_structural].

### P_imported — Imported External Theorem

The theorem uses an external result from mathematics or physics
(Gleason 1957, Lovelock 1971, Coleman-Mandula 1967, Noether 1918,
Weinberg-Witten 1980, etc.) whose hypotheses are verified from [P]
theorems. The external result itself is not derived from A1 — it is
imported as established mathematics.

**Example:** T9_grav — the Einstein field equations are the unique
rank-2 divergence-free tensor equations in 4D (Lovelock 1971). The
hypotheses (4D, Lorentzian, divergence-free) are derived from
Delta_signature [P], T8 [P], and the Bianchi identity.

### W — Witness

A numerical verification, not a proof. The theorem computes a
specific quantity and checks that it matches expectations. Useful
for sanity checks and cross-validation, but does not constitute
a logical derivation.

**Example:** Numerical eigenvalue checks in T_mass_ratios that verify
computed mass ratios lie within experimental bounds.

### O — Open

The theorem is claimed but the proof is incomplete or the result
has known gaps. Included for completeness and to flag where further
work is needed.

## Compound Tags

Some theorems carry compound tags like `P/P_structural`, meaning
the derivation contains both fully proved and structurally argued
components.

## Imported Theorems

When a [P] or [P_imported] theorem uses an external result, the
result dict includes an `imported_theorems` field listing:

```python
'imported_theorems': {
    'Gleason_1957': {
        'statement': 'Frame functions on H (dim≥3) are of the form Tr(ρ·P)',
        'our_use': 'Derives Born rule from A1 capacity measure'
    }
}
```

This makes the chain of reasoning auditable: you can check that
the hypotheses of the imported theorem are actually satisfied by
the framework's derived structure.

## Tag Distribution (v5.0)

| Tag | Count | Fraction |
|-----|-------|----------|
| P | ~95 | 74% |
| P_structural | ~15 | 12% |
| P/P_structural | ~5 | 4% |
| AXIOM | 1 | <1% |
| POSTULATE | 2 | 2% |
| W | ~11 | 9% |

The majority of theorems are fully proved from the axiom chain.
The structural arguments cluster in cosmology (where modelling
choices are unavoidable) and in mass ratios (where the qualitative
hierarchy is proved but exact values depend on the Froggatt-Nielsen
parameter identification).
