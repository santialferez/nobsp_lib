"""
GPU-optimized NObSP decomposition functions.

These functions keep all computations on GPU using PyTorch tensors,
avoiding CPU-NumPy conversions for significant speedup.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import numpy as np
from .oblique import oblique_projection_beta


def decompose_beta_gpu(
    X: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    model: nn.Module,
    problem_type: str = 'regression',
    device: Optional[torch.device] = None,
    regularization: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-optimized beta decomposition using partial regression.
    
    Keeps all computations on GPU, avoiding NumPy conversions.
    Returns torch tensors instead of NumPy arrays.
    
    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Input features [n_samples, n_features]
    y_pred : torch.Tensor or np.ndarray  
        Model predictions
    model : nn.Module
        Neural network model with forward method returning (prob, X_trans, y_lin)
    problem_type : str
        'regression' or 'classification'
    device : torch.device, optional
        Device for computation (auto-detected if None)
    regularization : float
        Regularization parameter for numerical stability
        
    Returns
    -------
    coefficients : torch.Tensor
        Beta coefficients [hidden_size, n_features * n_outputs] on GPU
    contributions : torch.Tensor
        Feature contributions [n_samples, n_features, n_outputs] on GPU
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    # Convert to GPU tensors
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()
    
    X = X.to(device)
    y_pred = y_pred.to(device)
    
    # Get dimensions
    n_samples, n_features = X.shape
    n_outputs = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
    
    # Get hidden layer size from model
    with torch.no_grad():
        if problem_type == 'classification':
            _, X_trans, _ = model(X)
        else:  # regression
            _, X_trans = model(X)
    hidden_size = X_trans.shape[1]
    
    # Initialize storage on GPU
    Beta = torch.zeros(hidden_size, n_features * n_outputs, device=device)
    contributions = torch.zeros(n_samples, n_features, n_outputs, device=device)
    
    # Batch process features (keep on GPU)
    for i in range(n_features):
        # Create target and reference inputs
        X_target = torch.zeros_like(X)
        X_target[:, i] = X[:, i]
        
        X_reference = X.clone()
        X_reference[:, i] = 0
        
        # Get transformations (stay on GPU)
        with torch.no_grad():
            if problem_type == 'classification':
                _, X_target_sub, _ = model(X_target)
                _, X_reference_sub, _ = model(X_reference)
            else:  # regression
                _, X_target_sub = model(X_target)
                _, X_reference_sub = model(X_reference)
        
        # Center the subspaces
        X_target_sub_centered = X_target_sub - X_target_sub.mean(dim=0)
        X_reference_sub_centered = X_reference_sub - X_reference_sub.mean(dim=0)
        
        # Process outputs (vectorize where possible)
        for j in range(n_outputs):
            y_target = y_pred[:, j] if n_outputs > 1 else y_pred.squeeze()
            y_target_centered = y_target - y_target.mean()
            
            # Compute beta coefficients (stays on GPU)
            beta = oblique_projection_beta(
                X_target_sub_centered, 
                X_reference_sub_centered,
                y_target_centered, 
                device, 
                lambda_reg=regularization
            )
            
            # Store coefficients
            coef_idx = j * n_features + i
            Beta[:, coef_idx] = beta.squeeze()
            
            # Compute contributions (stay on GPU)
            contrib = X_target_sub @ beta.unsqueeze(1)
            contributions[:, i, j] = contrib.squeeze()
    
    return Beta, contributions


def decompose_alpha_gpu(
    X: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    model: nn.Module,
    problem_type: str = 'regression',
    device: Optional[torch.device] = None,
    regularization: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-optimized alpha decomposition that exactly reproduces basic method.
    
    MATHEMATICAL GUARANTEE: Alpha coefficients are computed to exactly reconstruct 
    the basic method's oblique projections, ensuring perfect correlation.
    
    Parameters same as decompose_beta_gpu.
    
    Returns
    -------
    coefficients : torch.Tensor
        Alpha coefficients on GPU [hidden_size, n_features * n_outputs]
    contributions : torch.Tensor
        Feature contributions on GPU [n_samples, n_features, n_outputs]
    """
    # Import oblique projection
    from .oblique import oblique_projection
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    # Convert to GPU tensors
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()
    
    X = X.to(device)
    y_pred = y_pred.to(device)
    
    # Get dimensions
    n_samples, n_features = X.shape
    n_outputs = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Get hidden layer size
    with torch.no_grad():
        if problem_type == 'classification':
            _, X_trans, _ = model(X)
        else:  # regression
            _, X_trans = model(X)
    hidden_size = X_trans.shape[1]
    
    # STEP 1: Compute basic oblique projections (stay on GPU)
    contributions_basic = torch.zeros(n_samples, n_features, n_outputs, device=device)
    
    for i in range(n_features):
        # Create target input (only feature i)
        X_target = torch.zeros_like(X)
        X_target[:, i] = X[:, i]
        
        # Create reference input (all features except i)
        X_reference = X.clone()
        X_reference[:, i] = 0
        
        # Get transformations
        with torch.no_grad():
            if problem_type == 'classification':
                _, X_target_sub, _ = model(X_target)
                _, X_reference_sub, _ = model(X_reference)
            else:  # regression
                _, X_target_sub = model(X_target)
                _, X_reference_sub = model(X_reference)
        
        # Center the subspaces
        X_target_sub_centered = X_target_sub - X_target_sub.mean(dim=0)
        X_reference_sub_centered = X_reference_sub - X_reference_sub.mean(dim=0)
        
        # Compute oblique projection
        P_xy = oblique_projection(X_target_sub_centered, X_reference_sub_centered, device)
        
        # Apply projection to get contributions for all outputs
        for j in range(n_outputs):
            y_centered = y_pred[:, j] - y_pred[:, j].mean()
            y_e = P_xy @ y_centered
            contributions_basic[:, i, j] = y_e
    
    # STEP 2: Compute alpha coefficients that EXACTLY reconstruct basic projections
    Alpha = torch.zeros(hidden_size, n_features * n_outputs, device=device)
    Z_features = torch.zeros(n_samples, hidden_size, n_features, device=device)
    
    # Collect feature transformations and compute alpha coefficients
    for l in range(n_outputs):
        for i in range(n_features):
            # Create target input (only feature i)
            X_target = torch.zeros_like(X)
            X_target[:, i] = X[:, i]
            
            # Get transformation
            with torch.no_grad():
                if problem_type == 'classification':
                    _, X_target_sub, _ = model(X_target)
                else:  # regression
                    _, X_target_sub = model(X_target)
            
            # Store uncentered feature-specific transformation
            Z_features[:, :, i] = X_target_sub
            
            # Center the transformation
            X_target_sub_centered = X_target_sub - X_target_sub.mean(dim=0)
            
            # Target: basic method's projections for this feature-output combination
            y_target_basic = contributions_basic[:, i, l].reshape(-1, 1)
            
            # Compute alpha coefficients to reconstruct basic projections
            try:
                # Check for rank deficiency
                rank = torch.linalg.matrix_rank(X_target_sub_centered)
                if rank < min(X_target_sub_centered.shape):
                    # Use regularized least squares
                    XtX = X_target_sub_centered.T @ X_target_sub_centered
                    XtX_reg = XtX + regularization * torch.eye(XtX.shape[0], device=device)
                    Xty = X_target_sub_centered.T @ y_target_basic.view(-1)
                    Alpha[:, l*n_features+i] = torch.linalg.solve(XtX_reg, Xty).squeeze()
                else:
                    # Use standard least squares
                    Alpha[:, l*n_features+i] = torch.linalg.lstsq(
                        X_target_sub_centered, 
                        y_target_basic.view(-1, 1),
                        rcond=None, 
                        driver='gels'
                    )[0].squeeze()
            except:
                # Fallback to regularized solution
                XtX = X_target_sub_centered.T @ X_target_sub_centered
                XtX_reg = XtX + regularization * torch.eye(XtX.shape[0], device=device)
                Xty = X_target_sub_centered.T @ y_target_basic.view(-1)
                Alpha[:, l*n_features+i] = torch.linalg.solve(XtX_reg, Xty).squeeze()
    
    # STEP 3: Compute alpha contributions (should exactly match basic method)
    contributions = torch.zeros(n_samples, n_features, n_outputs, device=device)
    for l in range(n_outputs):
        for i in range(n_features):
            contributions[:, i, l] = (Z_features[:, :, i] @ Alpha[:, l*n_features+i:l*n_features+i+1]).squeeze()
    
    return Alpha, contributions

def decompose_beta_gpu_batched(
    X: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    model: nn.Module,
    problem_type: str = 'regression',
    device: Optional[torch.device] = None,
    regularization: float = 1e-6,
    batch_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-optimized beta decomposition with feature batching.
    
    Processes multiple features in parallel for better GPU utilization.
    
    Additional Parameters
    ---------------------
    batch_size : int
        Number of features to process in parallel
        
    Returns same as decompose_beta_gpu
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    # Convert to GPU tensors
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()
    
    X = X.to(device)
    y_pred = y_pred.to(device)
    
    # Get dimensions
    n_samples, n_features = X.shape
    n_outputs = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
    
    # Get hidden layer size
    with torch.no_grad():
        if problem_type == 'classification':
            _, X_trans, _ = model(X)
        else:  # regression
            _, X_trans = model(X)
    hidden_size = X_trans.shape[1]
    
    # Initialize storage on GPU
    Beta = torch.zeros(hidden_size, n_features * n_outputs, device=device)
    contributions = torch.zeros(n_samples, n_features, n_outputs, device=device)
    
    # Process features in batches
    for batch_start in range(0, n_features, batch_size):
        batch_end = min(batch_start + batch_size, n_features)
        batch_features = range(batch_start, batch_end)
        batch_size_actual = batch_end - batch_start
        
        # Create batched inputs [batch_size, n_samples, n_features]
        X_target_batch = torch.zeros(batch_size_actual, n_samples, n_features, device=device)
        X_reference_batch = X.unsqueeze(0).repeat(batch_size_actual, 1, 1)
        
        for idx, i in enumerate(batch_features):
            X_target_batch[idx, :, i] = X[:, i]
            X_reference_batch[idx, :, i] = 0
        
        # Reshape for batch processing
        X_target_flat = X_target_batch.view(-1, n_features)
        X_reference_flat = X_reference_batch.view(-1, n_features)
        
        # Get transformations for batch
        with torch.no_grad():
            if problem_type == 'classification':
                _, X_target_sub_flat, _ = model(X_target_flat)
                _, X_reference_sub_flat, _ = model(X_reference_flat)
            else:  # regression
                _, X_target_sub_flat = model(X_target_flat)
                _, X_reference_sub_flat = model(X_reference_flat)
        
        # Reshape back
        X_target_sub = X_target_sub_flat.view(batch_size_actual, n_samples, hidden_size)
        X_reference_sub = X_reference_sub_flat.view(batch_size_actual, n_samples, hidden_size)
        
        # Process each feature in batch
        for idx, i in enumerate(batch_features):
            X_target_sub_i = X_target_sub[idx]
            X_reference_sub_i = X_reference_sub[idx]
            
            # Center
            X_target_centered = X_target_sub_i - X_target_sub_i.mean(dim=0)
            X_reference_centered = X_reference_sub_i - X_reference_sub_i.mean(dim=0)
            
            # Process outputs
            for j in range(n_outputs):
                y_target = y_pred[:, j] if n_outputs > 1 else y_pred.squeeze()
                y_target_centered = y_target - y_target.mean()
                
                # Compute beta
                beta = oblique_projection_beta(
                    X_target_centered,
                    X_reference_centered,
                    y_target_centered,
                    device,
                    lambda_reg=regularization
                )
                
                # Store
                coef_idx = j * n_features + i
                Beta[:, coef_idx] = beta.squeeze()
                
                # Contributions
                contrib = X_target_sub_i @ beta.unsqueeze(1)
                contributions[:, i, j] = contrib.squeeze()
    
    return Beta, contributions