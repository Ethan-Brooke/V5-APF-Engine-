#!/usr/bin/env python3
"""Run the full APF theorem bank. Equivalent to: python -m apf.bank"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from apf.bank import main
main()
