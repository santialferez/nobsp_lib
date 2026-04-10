"""
Core NObSP algorithms and mathematical operations.

This module contains the core implementations of:
- Decomposition algorithms (decompose.py)
- GPU-optimized decomposition (decompose_gpu.py)
- Oblique projection operations (oblique.py)
- NObSP-CAM for CNN interpretability (nobsp_cam.py)
"""

# Import the CAM module
from .nobsp_cam import NObSPCAM

# Note: The old decomposition.py functions have been moved to decomposition.py.old
# The new API is available in nobsp.decomposition.NObSP class