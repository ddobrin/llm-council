"""LLM Council entry point for ADK CLI (`adk run agents/council`, `adk web`).

Exposes `root_agent` conforming to ADK application convention.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when invoked via adk run / adk web
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.agents.pipeline import build_council_workflow

root_agent = build_council_workflow()
