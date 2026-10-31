"""shopper_intent: session-level purchase intention model.

The package is organised as a small pipeline: fetch -> validate -> features ->
train -> evaluate -> serve -> drift, driven by the ``shopper_intent`` CLI.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
