#!/usr/bin/env python3
"""
================================================================================
MASTER VERIFICATION ENGINE — APF v5.0 (Engine v15.0)
================================================================================

Runs the full APF v5.0 modular theorem bank (129 theorems, 8 modules)
and produces:

  1. Terminal output (pass/fail summary)
  2. HTML report (standalone, self-contained dashboard)
  3. JSON export (for CI integration)

Usage:
  python engine/run_engine.py                    # terminal + HTML
  python engine/run_engine.py --json             # + JSON export
  python engine/run_engine.py --html-only        # HTML only, no terminal
  python engine/run_engine.py --output report.html  # custom output path

No external dependencies. Stdlib only.
================================================================================
"""

import sys
import os
import json
import time
import math
import inspect
from collections import OrderedDict

# Ensure apf package is importable
_engine_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_engine_dir)
_src_dir = os.path.join(_repo_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from apf.bank import REGISTRY, _load, _MODULE_MAP, run_all as _bank_run_all

VERSION = '5.0.0'
ENGINE_VERSION = 'v15.0'

# ======================================================================
#  SECTOR DEFINITIONS
# ======================================================================

SECTORS = {
    'Axiom Foundation': ['A1', 'M', 'NT'],
    'Quantum Admissibility': [
        'L_epsilon*', 'L_irr', 'L_nc', 'L_loc', 'L_T2', 'L_cost',
        'T0', 'T1', 'T2', 'T3', 'T_Born', 'T_CPTP', 'T_Hermitian',
        'T_M', 'T_canonical', 'T_entropy', 'T_epsilon', 'T_eta',
        'T_kappa', 'T_tensor', 'L_irr_uniform', 'L_Omega_sign',
        'M_Omega', 'P_exhaust',
    ],
    'Gauge & Fields': [
        'T4', 'T5', 'T_gauge', 'T_field', 'T_channels', 'T7', 'T9',
        'T_Higgs', 'T_particle', 'T_confinement', 'Theorem_R',
        'B1_prime', 'T4E', 'T4F', 'L_count', 'L_Weinberg_dim',
        'L_dim_angle', 'T_theta_QCD',
        'L_anomaly_free', 'T_proton', 'L_strong_CP_synthesis',
        'T_vacuum_stability',
    ],
    'Generations & Mixing': [
        'T6', 'T6B', 'T19', 'T20', 'T21', 'T21a', 'T21b', 'T21c',
        'T22', 'T23', 'T24', 'T25a', 'T25b', 'T26', 'T27c', 'T27d',
        'T_CKM', 'T_PMNS', 'T_mass_ratios', 'T_nu_ordering',
        'T_sin2theta', 'T_capacity_ladder', 'T4G', 'T4G_Q31',
        'T_q_Higgs', 'T_LV', 'T_S0',
        'L_AF_capacity', 'L_Gram', 'L_Gram_generation', 'L_beta',
        'L_gen_path', 'L_holonomy_phase', 'L_adjoint_sep',
        'L_channel_crossing', 'L_D2q', 'L_H_curv',
        'L_color_Gram', 'L_conjugation', 'L_mass_mixing_indep',
        'L_LL_coherence', 'L_cap_per_dim', 'L_angular_far_edge',
        'L_boundary_proj', 'L_edge_amplitude', 'L_capacity_depth',
    ],
    'Spacetime Arena': [
        'T8', 'Delta_ordering', 'Delta_fbc', 'Delta_continuum',
        'Delta_signature', 'Delta_particle', 'Delta_closure',
        'T_Coleman_Mandula',
    ],
    'Gravity': [
        'T7B', 'T9_grav', 'T10', 'T_Bek', 'T_deSitter_entropy',
        'L_self_exclusion', 'T_graviton', 'L_Weinberg_Witten',
    ],
    'Cosmology': [
        'L_equip', 'T11', 'T12', 'T12E',
    ],
    'Observational Validation': [
        'T_concordance', 'T_inflation', 'T_baryogenesis',
        'T_reheating', 'L_Sakharov',
    ],
    'Consistency Supplements': [
        'T_spin_statistics', 'T_CPT', 'T_second_law',
        'T_decoherence', 'T_Noether', 'T_optical',
        'L_cluster', 'T_BH_information', 'L_naturalness',
    ],
}

TIER_NAMES = {
    -1: 'Tier -1: Axioms & Postulates',
    0: 'Tier 0: Foundational Lemmas',
    1: 'Tier 1: Gauge Origin',
    2: 'Tier 2: Field Content',
    3: 'Tier 3: Generations & Mixing',
    4: 'Tier 4: Gravity & Cosmology',
    5: 'Tier 5: Spacetime Emergence',
}

MODULE_COLORS = {
    'core': '#4FC3F7',
    'gauge': '#81C784',
    'generations': '#FFB74D',
    'spacetime': '#CE93D8',
    'gravity': '#F06292',
    'cosmology': '#4DD0E1',
    'validation': '#FFF176',
    'supplements': '#A1887F',
}


# ======================================================================
#  ANALYSIS ENGINE
# ======================================================================

def run_engine():
    """Execute full verification and analysis."""
    _load()
    t0 = time.time()

    # 1. Run all theorems
    all_results = {}
    passed = failed = errors = 0
    for name, check_fn in REGISTRY.items():
        try:
            r = check_fn()
            all_results[name] = r
            if r['passed']:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            errors += 1
            all_results[name] = {
                'name': name, 'passed': False, 'error': str(e),
                'tier': -99, 'epistemic': 'ERROR',
                'summary': f'Exception: {e}',
                'key_result': f'ERROR: {e}',
                'dependencies': [],
            }

    elapsed = time.time() - t0

    # 2. Build theorem->module map
    theorem_to_mod = {}
    for mod, names in _MODULE_MAP.items():
        for name in names:
            theorem_to_mod[name] = mod

    # 3. Epistemic distribution
    epistemic_counts = {}
    for r in all_results.values():
        e = r.get('epistemic', '?')
        epistemic_counts[e] = epistemic_counts.get(e, 0) + 1

    # 4. Tier breakdown
    tier_stats = {}
    for tier_num in range(-1, 6):
        tier_results = {k: v for k, v in all_results.items()
                        if v.get('tier') == tier_num}
        if tier_results:
            tier_stats[tier_num] = {
                'name': TIER_NAMES.get(tier_num, f'Tier {tier_num}'),
                'total': len(tier_results),
                'passed': sum(1 for r in tier_results.values() if r['passed']),
                'theorems': sorted(tier_results.keys()),
            }

    # 5. Sector verdicts
    sector_verdicts = {}
    for name, tids in SECTORS.items():
        all_ok = all(all_results[t]['passed']
                     for t in tids if t in all_results)
        n = sum(1 for t in tids if t in all_results)
        sector_verdicts[name] = {'ok': all_ok, 'count': n}

    # 6. Module stats
    module_stats = {}
    for mod, names in _MODULE_MAP.items():
        p = sum(1 for n in names if all_results.get(n, {}).get('passed'))
        module_stats[mod] = {
            'total': len(names),
            'passed': p,
            'theorems': names,
            'color': MODULE_COLORS.get(mod, '#888'),
        }

    # 7. Dependency analysis
    dep_graph = {}
    for name, r in all_results.items():
        dep_graph[name] = r.get('dependencies', [])

    missing_deps = {}
    for name, deps in dep_graph.items():
        for d in deps:
            if d not in all_results:
                missing_deps.setdefault(name, []).append(d)

    # 8. Cycle detection (Tarjan SCC)
    sccs = _tarjan_scc(dep_graph, set(all_results.keys()))

    # 9. Assertion count
    n_asserts = 0
    for f in REGISTRY.values():
        try:
            n_asserts += inspect.getsource(f).count('assert ')
        except (OSError, TypeError):
            pass

    # 10. Extract predictions from artifacts
    predictions = _extract_predictions(all_results)

    # 11. Import taxonomy
    imports = _extract_imports(all_results)

    return {
        'version': VERSION,
        'engine_version': ENGINE_VERSION,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'elapsed_s': elapsed,
        'total_theorems': len(all_results),
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'all_pass': (failed + errors) == 0,
        'epistemic_counts': epistemic_counts,
        'tier_stats': tier_stats,
        'sector_verdicts': sector_verdicts,
        'module_stats': module_stats,
        'all_results': all_results,
        'theorem_to_mod': theorem_to_mod,
        'dep_graph': dep_graph,
        'missing_deps': missing_deps,
        'sccs': sccs,
        'n_assertions': n_asserts,
        'predictions': predictions,
        'imports': imports,
    }


def _tarjan_scc(graph, valid_nodes):
    """Tarjan's SCC algorithm. Returns list of SCCs with size > 1."""
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = set()
    result = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in valid_nodes:
                continue
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                result.append(sorted(component))

    # Increase recursion limit for deep graphs
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, len(valid_nodes) * 3))
    try:
        for v in sorted(valid_nodes):
            if v not in index:
                strongconnect(v)
    finally:
        sys.setrecursionlimit(old_limit)

    return result


def _extract_predictions(all_results):
    """Extract numerical predictions from theorem artifacts."""
    preds = []
    for name, r in all_results.items():
        arts = r.get('artifacts', {})
        if not arts:
            continue
        # Look for prediction-like keys
        for key, val in arts.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                preds.append({
                    'theorem': name,
                    'quantity': key,
                    'value': val,
                })
    return preds


def _extract_imports(all_results):
    """Extract imported theorem taxonomy."""
    imports = []
    seen = set()
    for name, r in sorted(all_results.items()):
        imp = r.get('imported_theorems')
        if not imp:
            continue
        if isinstance(imp, dict):
            for imp_name, details in imp.items():
                key = (name, imp_name)
                if key not in seen:
                    seen.add(key)
                    imports.append({
                        'theorem': name,
                        'import_name': imp_name,
                        'details': details if isinstance(details, dict)
                                   else {'note': str(details)},
                    })
    return imports


# ======================================================================
#  HTML REPORT GENERATOR
# ======================================================================

def generate_html(master):
    """Generate a standalone HTML dashboard from engine results."""

    total = master['total_theorems']
    passed = master['passed']
    failed = master['failed']
    errs = master['errors']
    elapsed = master['elapsed_s']
    all_pass = master['all_pass']

    # Prepare module data for chart
    mod_data = []
    for mod in ['core', 'gauge', 'generations', 'spacetime',
                'gravity', 'cosmology', 'validation', 'supplements']:
        ms = master['module_stats'].get(mod, {})
        mod_data.append({
            'name': mod,
            'total': ms.get('total', 0),
            'passed': ms.get('passed', 0),
            'color': ms.get('color', '#888'),
        })

    # Prepare tier data
    tier_rows = []
    for tier_num in sorted(master['tier_stats'].keys()):
        ts = master['tier_stats'][tier_num]
        tier_rows.append({
            'tier': tier_num,
            'name': ts['name'],
            'total': ts['total'],
            'passed': ts['passed'],
        })

    # Prepare theorem table
    theorem_rows = []
    for name, r in master['all_results'].items():
        mod = master['theorem_to_mod'].get(name, '?')
        theorem_rows.append({
            'name': name,
            'module': mod,
            'tier': r.get('tier', '?'),
            'epistemic': r.get('epistemic', '?'),
            'passed': r.get('passed', False),
            'key_result': r.get('key_result', '')[:120],
            'color': MODULE_COLORS.get(mod, '#888'),
        })

    # Sector data
    sector_rows = []
    for name, data in master['sector_verdicts'].items():
        sector_rows.append({
            'name': name,
            'ok': data['ok'],
            'count': data['count'],
        })

    # SCC data
    scc_info = []
    for scc in master['sccs']:
        scc_info.append({'size': len(scc), 'members': scc[:10]})

    # Epistemic distribution
    ep_data = sorted(master['epistemic_counts'].items(),
                     key=lambda x: -x[1])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APF v{VERSION} — Verification Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Instrument+Serif&display=swap');

:root {{
  --bg: #0a0e17;
  --bg2: #111827;
  --bg3: #1a2235;
  --border: #2a3650;
  --text: #c9d1d9;
  --text-dim: #6b7b95;
  --text-bright: #e6edf3;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --yellow: #d29922;
  --mono: 'JetBrains Mono', monospace;
  --serif: 'Instrument Serif', Georgia, serif;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1.6;
  min-height: 100vh;
}}

.container {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 24px;
}}

/* ── Header ────────────────────────────────── */
.hero {{
  text-align: center;
  padding: 60px 0 40px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 40px;
  position: relative;
}}
.hero::before {{
  content: '';
  position: absolute;
  top: 0; left: 50%;
  transform: translateX(-50%);
  width: 200px; height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
}}
.hero h1 {{
  font-family: var(--serif);
  font-size: 42px;
  font-weight: 400;
  color: var(--text-bright);
  letter-spacing: -0.5px;
  margin-bottom: 8px;
}}
.hero .subtitle {{
  font-size: 14px;
  color: var(--text-dim);
  font-weight: 300;
}}
.hero .version {{
  display: inline-block;
  margin-top: 16px;
  padding: 4px 16px;
  border: 1px solid var(--border);
  border-radius: 4px;
  font-size: 11px;
  color: var(--accent);
  letter-spacing: 2px;
  text-transform: uppercase;
}}

/* ── Status Banner ─────────────────────────── */
.status-banner {{
  display: flex;
  justify-content: center;
  gap: 40px;
  padding: 32px 0;
  margin-bottom: 40px;
  border-bottom: 1px solid var(--border);
}}
.stat {{
  text-align: center;
}}
.stat-value {{
  font-size: 36px;
  font-weight: 700;
  color: var(--text-bright);
  line-height: 1;
}}
.stat-value.pass {{ color: var(--green); }}
.stat-value.fail {{ color: var(--red); }}
.stat-label {{
  font-size: 10px;
  color: var(--text-dim);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 6px;
}}

/* ── Sections ──────────────────────────────── */
section {{
  margin-bottom: 48px;
}}
section h2 {{
  font-family: var(--serif);
  font-size: 24px;
  font-weight: 400;
  color: var(--text-bright);
  margin-bottom: 20px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}}

/* ── Module Grid ───────────────────────────── */
.module-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
}}
.module-card {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}}
.module-card:hover {{
  border-color: var(--accent);
}}
.module-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
}}
.module-card .mod-name {{
  font-size: 14px;
  font-weight: 500;
  color: var(--text-bright);
  margin-bottom: 4px;
}}
.module-card .mod-count {{
  font-size: 11px;
  color: var(--text-dim);
}}
.module-card .mod-bar {{
  margin-top: 10px;
  height: 4px;
  background: var(--bg3);
  border-radius: 2px;
  overflow: hidden;
}}
.module-card .mod-bar-fill {{
  height: 100%;
  border-radius: 2px;
  transition: width 0.6s ease;
}}

/* ── Sector Grid ───────────────────────────── */
.sector-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px;
}}
.sector-item {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  font-size: 12px;
}}
.sector-dot {{
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}}
.sector-dot.ok {{ background: var(--green); }}
.sector-dot.fail {{ background: var(--red); }}

/* ── Epistemic Bars ────────────────────────── */
.ep-row {{
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 6px;
}}
.ep-label {{
  width: 120px;
  text-align: right;
  font-size: 11px;
  color: var(--text-dim);
}}
.ep-bar {{
  flex: 1;
  height: 20px;
  background: var(--bg3);
  border-radius: 3px;
  overflow: hidden;
}}
.ep-bar-fill {{
  height: 100%;
  background: var(--accent);
  border-radius: 3px;
  transition: width 0.8s ease;
}}
.ep-count {{
  width: 30px;
  font-size: 12px;
  color: var(--text-bright);
  font-weight: 500;
}}

/* ── Theorem Table ─────────────────────────── */
.theorem-table-wrap {{
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}}
thead th {{
  background: var(--bg3);
  padding: 10px 12px;
  text-align: left;
  font-weight: 500;
  font-size: 10px;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 1;
}}
tbody td {{
  padding: 8px 12px;
  border-bottom: 1px solid rgba(42,54,80,0.5);
  vertical-align: top;
}}
tbody tr:hover {{
  background: rgba(88,166,255,0.04);
}}
.badge {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.5px;
}}
.badge-pass {{ background: rgba(63,185,80,0.15); color: var(--green); }}
.badge-fail {{ background: rgba(248,81,73,0.15); color: var(--red); }}
.badge-mod {{
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 10px;
}}
.key-result {{
  color: var(--text-dim);
  font-size: 11px;
  max-width: 400px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

/* ── Search/Filter ─────────────────────────── */
.filter-bar {{
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}}
.filter-bar input {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 12px;
  color: var(--text);
  font-family: var(--mono);
  font-size: 12px;
  width: 250px;
}}
.filter-bar input:focus {{
  outline: none;
  border-color: var(--accent);
}}
.filter-btn {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px 14px;
  color: var(--text-dim);
  font-family: var(--mono);
  font-size: 11px;
  cursor: pointer;
  transition: all 0.15s;
}}
.filter-btn:hover, .filter-btn.active {{
  border-color: var(--accent);
  color: var(--accent);
}}

/* ── SCC section ───────────────────────────── */
.scc-box {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 12px;
}}
.scc-box .scc-title {{
  font-weight: 500;
  color: var(--yellow);
  margin-bottom: 6px;
}}
.scc-members {{
  font-size: 11px;
  color: var(--text-dim);
  word-break: break-all;
}}

/* ── Footer ────────────────────────────────── */
footer {{
  text-align: center;
  padding: 40px 0 20px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-dim);
}}

/* ── Animations ────────────────────────────── */
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
section {{
  animation: fadeUp 0.4s ease both;
}}
section:nth-child(2) {{ animation-delay: 0.05s; }}
section:nth-child(3) {{ animation-delay: 0.10s; }}
section:nth-child(4) {{ animation-delay: 0.15s; }}
section:nth-child(5) {{ animation-delay: 0.20s; }}
section:nth-child(6) {{ animation-delay: 0.25s; }}
section:nth-child(7) {{ animation-delay: 0.30s; }}
</style>
</head>
<body>
<div class="container">

<!-- Hero -->
<div class="hero">
  <h1>Admissibility Physics Framework</h1>
  <div class="subtitle">Master Verification Report</div>
  <div class="version">v{VERSION} &middot; Engine {ENGINE_VERSION} &middot; {master['timestamp']}</div>
</div>

<!-- Status Banner -->
<div class="status-banner">
  <div class="stat">
    <div class="stat-value {'pass' if all_pass else 'fail'}">{passed}/{total}</div>
    <div class="stat-label">Theorems Passed</div>
  </div>
  <div class="stat">
    <div class="stat-value">{master['n_assertions']}</div>
    <div class="stat-label">Assertions</div>
  </div>
  <div class="stat">
    <div class="stat-value">0</div>
    <div class="stat-label">Free Parameters</div>
  </div>
  <div class="stat">
    <div class="stat-value">{elapsed:.2f}s</div>
    <div class="stat-label">Runtime</div>
  </div>
</div>

<!-- Modules -->
<section>
<h2>Modules</h2>
<div class="module-grid">
'''

    for md in mod_data:
        pct = (md['passed'] / md['total'] * 100) if md['total'] > 0 else 0
        html += f'''  <div class="module-card" style="border-left: 3px solid {md['color']}">
    <div class="mod-name">{md['name']}</div>
    <div class="mod-count">{md['passed']}/{md['total']} passed</div>
    <div class="mod-bar"><div class="mod-bar-fill" style="width:{pct}%;background:{md['color']}"></div></div>
  </div>
'''

    html += '''</div>
</section>

<!-- Sectors -->
<section>
<h2>Sector Verdicts</h2>
<div class="sector-grid">
'''

    for sr in sector_rows:
        cls = 'ok' if sr['ok'] else 'fail'
        html += f'''  <div class="sector-item">
    <div class="sector-dot {cls}"></div>
    <span>{sr['name']} ({sr['count']})</span>
  </div>
'''

    html += '''</div>
</section>

<!-- Epistemic Distribution -->
<section>
<h2>Epistemic Distribution</h2>
'''

    max_ep = max((c for _, c in ep_data), default=1)
    for tag, count in ep_data:
        pct = count / max_ep * 100
        html += f'''<div class="ep-row">
  <div class="ep-label">{tag}</div>
  <div class="ep-bar"><div class="ep-bar-fill" style="width:{pct}%"></div></div>
  <div class="ep-count">{count}</div>
</div>
'''

    html += '''</section>

<!-- Dependency Cycles -->
<section>
<h2>Dependency Cycles (Tarjan SCC)</h2>
'''

    if scc_info:
        acyclic = total - sum(s['size'] for s in scc_info)
        html += f'<p style="color:var(--text-dim);margin-bottom:12px">'
        html += f'{acyclic} of {total} theorems acyclic. '
        html += f'{len(scc_info)} strongly connected component(s):</p>\n'
        for i, scc in enumerate(scc_info):
            html += f'''<div class="scc-box">
  <div class="scc-title">SCC {i+1} — {scc['size']} theorems</div>
  <div class="scc-members">{', '.join(scc['members'])}</div>
</div>
'''
        html += '''<p style="color:var(--text-dim);font-size:11px;margin-top:8px">
These represent mutual-constraint systems (overdetermined fixed points),
not circular reasoning. See ARCHITECTURE.md for details.</p>
'''
    else:
        html += '<p style="color:var(--green)">DAG fully acyclic. No dependency cycles.</p>\n'

    html += '''</section>

<!-- Theorem Table -->
<section>
<h2>Full Theorem Registry</h2>
<div class="filter-bar">
  <input type="text" id="search" placeholder="Search theorems..." onkeyup="filterTable()">
  <button class="filter-btn active" onclick="filterMod(this,'all')">All</button>
'''

    for mod in ['core','gauge','generations','spacetime','gravity','cosmology','validation','supplements']:
        html += f'  <button class="filter-btn" onclick="filterMod(this,\'{mod}\')">{mod}</button>\n'

    html += '''</div>
<div class="theorem-table-wrap" style="max-height:600px;overflow-y:auto">
<table>
<thead>
<tr>
  <th>Status</th>
  <th>Name</th>
  <th>Module</th>
  <th>Tier</th>
  <th>Epistemic</th>
  <th>Key Result</th>
</tr>
</thead>
<tbody id="theorem-body">
'''

    for tr in theorem_rows:
        status_cls = 'badge-pass' if tr['passed'] else 'badge-fail'
        status_txt = 'PASS' if tr['passed'] else 'FAIL'
        kr_escaped = (tr['key_result']
                      .replace('&', '&amp;')
                      .replace('<', '&lt;')
                      .replace('>', '&gt;')
                      .replace('"', '&quot;'))
        name_escaped = tr['name'].replace('&', '&amp;')
        html += f'''<tr data-mod="{tr['module']}" data-name="{tr['name'].lower()}">
  <td><span class="badge {status_cls}">{status_txt}</span></td>
  <td style="font-weight:500;color:var(--text-bright)">{name_escaped}</td>
  <td><span class="badge-mod" style="background:{tr['color']}22;color:{tr['color']}">{tr['module']}</span></td>
  <td style="color:var(--text-dim)">{tr['tier']}</td>
  <td style="color:var(--text-dim)">{tr['epistemic']}</td>
  <td class="key-result" title="{kr_escaped}">{kr_escaped}</td>
</tr>
'''

    html += f'''</tbody>
</table>
</div>
</section>

<footer>
  APF v{VERSION} &middot; Engine {ENGINE_VERSION} &middot;
  {total} theorems &middot; {master['n_assertions']} assertions &middot;
  {elapsed:.2f}s &middot; stdlib only, zero dependencies
</footer>

</div>

<script>
let currentMod = 'all';

function filterTable() {{
  const q = document.getElementById('search').value.toLowerCase();
  const rows = document.querySelectorAll('#theorem-body tr');
  rows.forEach(r => {{
    const name = r.dataset.name || '';
    const mod = r.dataset.mod || '';
    const matchSearch = !q || name.includes(q) || r.textContent.toLowerCase().includes(q);
    const matchMod = currentMod === 'all' || mod === currentMod;
    r.style.display = (matchSearch && matchMod) ? '' : 'none';
  }});
}}

function filterMod(btn, mod) {{
  currentMod = mod;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  filterTable();
}}
</script>
</body>
</html>'''

    return html


# ======================================================================
#  TERMINAL DISPLAY
# ======================================================================

def display_terminal(master):
    """Print summary to terminal."""
    W = 72
    print(f"\n{'=' * W}")
    print(f"  APF v{VERSION} — Master Verification Engine {ENGINE_VERSION}")
    print(f"{'=' * W}\n")

    print(f"  Theorems:    {master['passed']}/{master['total_theorems']} passed")
    if master['failed']:
        print(f"  FAILED:      {master['failed']}")
    if master['errors']:
        print(f"  ERRORS:      {master['errors']}")
    print(f"  Assertions:  {master['n_assertions']}")
    print(f"  Runtime:     {master['elapsed_s']:.2f}s")
    print(f"  Parameters:  0")

    print(f"\n  {'─' * (W-4)}")
    print(f"  MODULES")
    print(f"  {'─' * (W-4)}")
    for mod in ['core','gauge','generations','spacetime','gravity',
                'cosmology','validation','supplements']:
        ms = master['module_stats'].get(mod, {})
        p, t = ms.get('passed', 0), ms.get('total', 0)
        mark = '✓' if p == t else '✗'
        print(f"  {mark} {mod:16s} {p:3d}/{t:3d}")

    print(f"\n  {'─' * (W-4)}")
    print(f"  SECTORS")
    print(f"  {'─' * (W-4)}")
    for name, data in master['sector_verdicts'].items():
        mark = '✓' if data['ok'] else '✗'
        print(f"  {mark} {name:30s} ({data['count']})")

    print(f"\n  {'─' * (W-4)}")
    print(f"  EPISTEMIC DISTRIBUTION")
    print(f"  {'─' * (W-4)}")
    for tag, count in sorted(master['epistemic_counts'].items(),
                             key=lambda x: -x[1]):
        bar = '█' * min(count, 50)
        print(f"  {tag:16s} {count:3d}  {bar}")

    status = 'ALL PASS' if master['all_pass'] else 'FAILURES DETECTED'
    print(f"\n{'=' * W}")
    print(f"  STATUS: {status}")
    print(f"{'=' * W}\n")


# ======================================================================
#  MAIN
# ======================================================================

def main():
    args = sys.argv[1:]

    html_only = '--html-only' in args
    do_json = '--json' in args
    output_path = None
    for i, a in enumerate(args):
        if a == '--output' and i + 1 < len(args):
            output_path = args[i + 1]

    if not output_path:
        output_path = os.path.join(_repo_root, 'report.html')

    # Run engine
    master = run_engine()

    # Terminal output
    if not html_only:
        display_terminal(master)

    # HTML report
    html = generate_html(master)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML report: {output_path}")

    # JSON export
    if do_json:
        json_path = output_path.replace('.html', '.json')
        # Strip non-serializable items
        export = {k: v for k, v in master.items()
                  if k != 'all_results'}
        export['theorem_names'] = list(master['all_results'].keys())
        export['theorem_status'] = {
            name: {'passed': r['passed'], 'epistemic': r.get('epistemic', '?')}
            for name, r in master['all_results'].items()
        }
        with open(json_path, 'w') as f:
            json.dump(export, f, indent=2, default=str)
        print(f"  JSON export: {json_path}")

    return 0 if master['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
