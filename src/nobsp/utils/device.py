"""
Device management utilities for NObSP.

Handles automatic device detection and assignment.
"""

import torch


def auto_detect_device() -> torch.device:
    """
    Automatically detect the best available device.
    
    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Returns
    -------
    device : torch.device
        The detected device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_name(device: torch.device) -> str:
    """
    Get a human-readable name for the device.
    
    Parameters
    ----------
    device : torch.device
        The device
        
    Returns
    -------
    name : str
        Human-readable device name
    """
    if device.type == 'cuda':
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(device)})"
        else:
            return "CUDA (not available)"
    elif device.type == 'mps':
        return "Apple Silicon GPU (MPS)"
    else:
        return "CPU"


def ensure_device_consistency(*tensors, device=None):
    """
    Ensure all tensors are on the same device.
    
    Parameters
    ----------
    *tensors : torch.Tensor
        Tensors to check
    device : torch.device, optional
        Target device. If None, uses the device of the first tensor
        
    Returns
    -------
    tensors : tuple of torch.Tensor
        Tensors moved to the same device
    """
    if not tensors:
        return tensors
    
    # Determine target device
    if device is None:
        device = tensors[0].device
    
    # Move all tensors to target device
    return tuple(t.to(device) if torch.is_tensor(t) else t for t in tensors)