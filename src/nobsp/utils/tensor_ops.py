"""
Tensor operation utilities for NObSP.

Handles conversions between numpy arrays and PyTorch tensors.
"""

import torch
import numpy as np
from typing import Union, Optional


def to_tensor(
    array: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Convert array to PyTorch tensor.
    
    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        Input array
    device : torch.device, optional
        Target device
    dtype : torch.dtype, optional
        Target dtype (default: torch.float32)
        
    Returns
    -------
    tensor : torch.Tensor
        Converted tensor
    """
    if dtype is None:
        dtype = torch.float32
    
    if isinstance(array, torch.Tensor):
        tensor = array
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
    else:
        # Convert from numpy
        tensor = torch.from_numpy(array)
        if dtype is not None:
            tensor = tensor.to(dtype)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert tensor to numpy array.
    
    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        Input tensor
        
    Returns
    -------
    array : np.ndarray
        Converted array
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    
    # Move to CPU if necessary and convert
    if tensor.is_cuda or (hasattr(tensor, 'is_mps') and tensor.is_mps):
        tensor = tensor.cpu()
    
    return tensor.detach().numpy()


def ensure_tensor_shape(
    tensor: torch.Tensor,
    expected_dims: int,
    name: str = "tensor"
) -> torch.Tensor:
    """
    Ensure tensor has expected number of dimensions.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor
    expected_dims : int
        Expected number of dimensions
    name : str
        Name for error messages
        
    Returns
    -------
    tensor : torch.Tensor
        Tensor with correct dimensions
        
    Raises
    ------
    ValueError
        If tensor cannot be reshaped to expected dimensions
    """
    if tensor.ndim < expected_dims:
        # Try to add dimensions
        while tensor.ndim < expected_dims:
            tensor = tensor.unsqueeze(-1)
    elif tensor.ndim > expected_dims:
        raise ValueError(
            f"{name} has {tensor.ndim} dimensions, expected {expected_dims}"
        )
    
    return tensor


def safe_inverse(
    matrix: torch.Tensor,
    regularization: float = 1e-6
) -> torch.Tensor:
    """
    Compute a numerically stable matrix inverse.
    
    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix to invert
    regularization : float
        Regularization parameter for numerical stability
        
    Returns
    -------
    inverse : torch.Tensor
        Inverted matrix
    """
    n = matrix.shape[0]
    device = matrix.device
    
    # Add regularization to diagonal
    reg_matrix = matrix + regularization * torch.eye(n, device=device)
    
    try:
        # Try standard inverse
        return torch.linalg.inv(reg_matrix)
    except:
        # Fall back to pseudoinverse
        return torch.linalg.pinv(matrix)