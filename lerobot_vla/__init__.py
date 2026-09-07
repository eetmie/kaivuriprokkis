"""LeRobot data collection and VLA deployment for the MASI excavator.

The public workflows are :mod:`record_episodes` and :mod:`run_inference`.
Split-engine implementations are internal runtime details under
:mod:`lerobot_vla.runtime`.
"""

from __future__ import annotations

import importlib
import sys


def __getattr__(name: str):
    """Resolve former split-module attributes during the layout transition."""
    targets = {
        "smolvla_split": ".runtime.smolvla",
        "xvla_split": ".runtime.xvla",
    }
    if name not in targets:
        raise AttributeError(name)
    module = importlib.import_module(targets[name], __name__)
    globals()[name] = module
    sys.modules[f"{__name__}.{name}"] = module
    return module
