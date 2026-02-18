"""Shared result constructor and epistemic tag definitions.

Every theorem check function returns a dict produced by result().
Epistemic tags classify the proof status of each theorem.

APF v5.0
"""

__all__ = ['result', 'Epistemic']


class Epistemic:
    """Epistemic status tags for theorems.

    P              — Proved: follows from the axiom chain (A1 + lemmas).
    P_structural   — Structural argument: robust but involves modelling
                     choices (e.g., slow-roll approximation, perturbative
                     decay rate). The qualitative conclusion is [P]; the
                     specific numerical value is model-dependent.
    P_imported     — Uses an imported external theorem (Gleason, Lovelock,
                     Coleman-Mandula, Noether, etc.) whose hypotheses are
                     verified from [P] theorems.
    W              — Witness: numerical verification, not a proof.
    O              — Open: claimed but not yet proved.
    """
    P = 'P'
    P_structural = 'P_structural'
    P_imported = 'P_imported'
    W = 'W'
    O = 'O'


def result(name, tier, epistemic, summary, key_result,
           dependencies=None, passed=True, artifacts=None,
           imported_theorems=None, cross_refs=None):
    """Uniform result constructor for all theorems.

    Parameters
    ----------
    name : str
        Human-readable theorem name.
    tier : int
        Logical tier in the derivation hierarchy (-1 to 5).
    epistemic : str
        One of Epistemic.P, .P_structural, .P_imported, .W, .O.
    summary : str
        One-paragraph proof sketch or description.
    key_result : str
        One-line statement of the main result.
    dependencies : list[str], optional
        Names of theorems this one depends on.
    passed : bool
        Whether all assertions passed.
    artifacts : dict, optional
        Computed quantities, numerical witnesses, etc.
    imported_theorems : dict, optional
        External theorems used (name -> {statement, our_use}).
    cross_refs : list[str], optional
        Related theorems (informational, not dependencies).

    Returns
    -------
    dict
        Standardized result record.
    """
    r = {
        'name': name,
        'tier': tier,
        'passed': passed,
        'epistemic': epistemic.value if hasattr(epistemic, 'value')
                     else epistemic,
        'summary': summary,
        'key_result': key_result,
        'dependencies': dependencies or [],
        'cross_refs': cross_refs or [],
        'artifacts': artifacts or {},
    }
    if imported_theorems:
        r['imported_theorems'] = imported_theorems
    return r
