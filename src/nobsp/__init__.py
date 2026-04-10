"""
NObSP - Neural Oblique Subspace Projections for Interpretability

A unified framework for neural network interpretability providing:
- Tabular data decomposition via NObSP class
- CNN interpretability via NObSPVision class  
"""

# Import main APIs
from .decomposition import NObSP  # Tabular API
from .vision import NObSPVision   # Vision API

# Version
__version__ = '0.1.0'

# Main exports
__all__ = [
    'NObSP',        # Tabular decomposition
    'NObSPVision',  # Vision/CNN CAM generation
    '__version__'
]