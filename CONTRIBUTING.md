# Contributing & AI Review Guide

This repository is set up for AI-assisted code review. External AI tools
(and humans) can read, clone, and evaluate the code. Only the repository
owner can merge changes.

## How to Evaluate This Code

### 1. Clone and run

```bash
git clone https://github.com/Ethan-Brooke/V5-APF-Engine-.git
cd V5-APF-Engine-
```

### 2. Run the theorem bank (zero dependencies)

```bash
PYTHONPATH=src python -m apf.bank
```

Expected output: `129 passed, 0 failed, 0 errors, 129 total`

### 3. Run the test suite

```bash
pip install pytest
PYTHONPATH=src python -m pytest tests/ -v
```

### 4. Run a single module

```bash
PYTHONPATH=src python -m apf.bank --module core
PYTHONPATH=src python -m apf.bank --module gauge
PYTHONPATH=src python -m apf.bank --module generations
```

### 5. List all theorems

```bash
PYTHONPATH=src python -m apf.bank --list
```

## What to Check

- **All 129 theorems pass** — any failure is a regression.
- **No external dependencies** — only stdlib (`math`, `cmath`, `fractions`). Any import of numpy/scipy/etc. is a violation.
- **DAG integrity** — `tests/test_dependencies.py` verifies no circular dependencies between modules.
- **Epistemic tags are accurate** — each theorem's proof status tag (P, W, O, etc.) should match its actual derivation rigor.
- **Firewall modules are terminal** — `validation.py` and `supplements.py` must not be imported by other physics modules.

## Requirements

- Python >= 3.10
- No third-party packages (stdlib only)
- pytest (for test suite only, not for the theorem bank itself)

## Access Model

| Role | Can read | Can clone/fork | Can open PRs | Can merge |
|------|----------|---------------|-------------|-----------|
| Public / AI reviewer | Yes | Yes | Yes | No |
| Repository owner | Yes | Yes | Yes | Yes |

PRs require CI to pass and owner approval before merging.
