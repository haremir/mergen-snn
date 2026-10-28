"""Mergen SNN Cython Hızlandırma Modülleri."""

try:
    from mergen.cython_ext.fast_lif import (
        fast_lif_forward,
        fast_lif_sequence_backward,
        fast_lif_sequence_forward,
    )
except ImportError:
    fast_lif_forward = None
    fast_lif_sequence_forward = None
    fast_lif_sequence_backward = None

try:
    from mergen.cython_ext.fast_gemm import fast_gemm_forward
except ImportError:
    fast_gemm_forward = None

__all__ = [
    "fast_lif_forward",
    "fast_lif_sequence_forward",
    "fast_lif_sequence_backward",
    "fast_gemm_forward",
]
