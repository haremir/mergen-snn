"""Mergen SNN Cython Hızlandırma Modülleri."""

try:
    from mergen.cython_ext.fast_lif import fast_lif_forward
except ImportError:
    fast_lif_forward = None

__all__ = ["fast_lif_forward"]
