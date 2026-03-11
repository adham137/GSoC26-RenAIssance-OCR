"""
conftest.py
===========
Root-level pytest configuration for the Agentic OCR Framework.

Placing conftest.py at the PROJECT ROOT (not inside test/ or tests/)
ensures pytest always adds the project root to sys.path, regardless
of which directory the test files live in or where pytest is invoked.

This is the canonical fix for:
    ModuleNotFoundError: No module named 'src.orchestrator.agentic_orchestrator'
"""

import sys
from pathlib import Path

# This file sits at the project root, so Path(__file__).parent IS the root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

