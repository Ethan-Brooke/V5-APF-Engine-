# Admissibility Physics Engine

Standalone artifact implementing the enforcement potential, capacity
calculations, and particle spectrum derivation. Included for reference
alongside the theorem bank.

This engine is **not** part of the theorem bank's verification pipeline.
It is a separate computational tool that explores the framework's
predictions interactively.

## Usage

```bash
python admissibility_physics_engine.py
```

## Relationship to Theorem Bank

The engine implements many of the same calculations verified by the
theorem bank, but in a different format — as a continuous computation
rather than pass/fail checks. The theorem bank is the authoritative
verification; the engine is an exploratory tool.
