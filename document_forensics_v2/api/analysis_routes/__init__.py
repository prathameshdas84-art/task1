"""PDF analysis endpoints: /analyze, /annotated-image, /hidden-text.

The router is assembled here — importing the endpoint modules registers
their routes on the shared APIRouter instance."""

from .base import router
from . import analyze, render, hidden_text  # noqa: E402,F401  (route registration)

__all__ = ["router"]
