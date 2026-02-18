"""Admissibility Physics Framework (APF) v5.0

An executable theorem bank deriving the Standard Model of particle
physics and cosmological observables from a single axiom: finite
enforceability of distinguishable records (A1).

Usage
-----
    from apf import REGISTRY, run_all
    results = run_all()                    # run all 129 theorems
    results = run_all(modules=['core'])    # run core only

CLI
---
    python -m apf.bank                     # run all
    python -m apf.bank --module gravity    # run one module

License: MIT
"""

__version__ = '5.0.0'

from apf.bank import REGISTRY, run_all

__all__ = ['REGISTRY', 'run_all', '__version__']
