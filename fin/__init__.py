"""
fin - Python bindings for f5c nanopore event alignment with CPU/GPU acceleration.
"""

from .core.eventalign import EventAlign

__version__ = "0.1.0"
__all__ = ["EventAlign"]