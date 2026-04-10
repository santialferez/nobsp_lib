"""
Core oblique projection functions.

These are the mathematical foundations of the NObSP method.
"""

import torch
import numpy as np
from typing import Optional, Union


def oblique_projection(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute the oblique projection onto X along Y.
    
    If Q is the orthogonal complement of the projection onto Y along X,
    then the oblique projection onto X along Y is defined by:
    P_{X/Y} = X(X'QX)^{-1}X'Q
    
    Parameters
    ----------
    X : torch.Tensor
        Matrix onto which we perform the oblique projection
        
    Y : torch.Tensor
        Matrix along which we project
        
    device : torch.device, optional
        Device for computation
        
    Returns
    -------
    P_xy : torch.Tensor
        The oblique projection matrix
    """
    # Auto-detect device if not provided
    if device is None:
        device = X.device if torch.is_tensor(X) else torch.device('cpu')
    
    # Ensure tensors are on the correct device
    X = X.to(device)
    Y = Y.to(device)
    
    # Computing the orthogonal projection matrix onto the subspace given by Y
    P = Y @ torch.linalg.pinv(torch.t(Y) @ Y) @ torch.t(Y)
    
    # Computing the complement of P
    Q = torch.eye(Y.shape[0], device=device) - P
    
    # Computing the oblique projection matrix onto X along Y
    # Fix: Q is symmetric, so torch.t(Q) should just be Q
    P_xy = X @ torch.linalg.pinv(torch.t(X) @ Q @ X) @ torch.t(X) @ Q
    
    return P_xy


def oblique_projection_beta(
    X_k: torch.Tensor,
    X_nk: torch.Tensor,
    y: torch.Tensor,
    device: Optional[torch.device] = None,
    lambda_reg: float = 1e-4
) -> torch.Tensor:
    """
    Compute beta coefficients using partial regression approach.
    
    This function implements an alternative to the Q-space projection approach
    by using partial regression to find the contribution of feature k.
    
    Algorithm:
    1. Regress y on X_nk to get residuals y_k
    2. Regress X_k on X_nk to get residuals A_k  
    3. Regress y_k on A_k to get beta coefficients
    
    Parameters
    ----------
    X_k : torch.Tensor
        The feature-specific transformation matrix (target subspace)
        
    X_nk : torch.Tensor
        The reference subspace transformation matrix (all features except k)
        
    y : torch.Tensor
        The target output vector
        
    device : torch.device, optional
        Device to perform computations on
        
    lambda_reg : float, default=1e-4
        Regularization parameter for numerical stabilityWEB
        
    Returns
    -------Lis
    beta : torch.Tensor
        The beta coefficients for feature k
    """
    # Auto-detect device if not provided
    if device is None:
        device = X_k.device if torch.is_tensor(X_k) else torch.device('cpu')
    
    # Ensure all tensors are on the correct device
    X_k = X_k.to(device) # target subspace
    X_nk = X_nk.to(device) # reference subspace
    y = y.to(device) # target output
    
    # Check dimensions and add small regularization for numerical stability
    eps = 1e-8
    
    # Check if X_nk is empty (happens when all other features are zero)
    if X_nk.shape[1] == 0 or torch.linalg.matrix_rank(X_nk) == 0:
        # If no reference subspace, beta is just direct regression
        beta = torch.linalg.lstsq(X_k + lambda_reg * torch.eye(X_k.shape[0], device=device), y, rcond=None)[0]
        return beta
    
    # Regress y on X_nk to get residuals
    try:
        # Use regularized least squares for stability
        XtX = torch.t(X_nk) @ X_nk
        XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0], device=device)
        alpha_y = torch.linalg.solve(XtX_reg, torch.t(X_nk) @ y)
        yk = y - X_nk @ alpha_y
    except:
        # Fallback to pseudoinverse
        alpha_y = torch.linalg.pinv(X_nk) @ y
        yk = y - X_nk @ alpha_y
    
    # Regress X_k on X_nk to get residuals
    try:
        alpha_x = torch.linalg.solve(XtX_reg, torch.t(X_nk) @ X_k)
        Ak = X_k - X_nk @ alpha_x
    except:
        # Fallback to pseudoinverse
        alpha_x = torch.linalg.pinv(X_nk) @ X_k
        Ak = X_k - X_nk @ alpha_x
    
    # Check if Ak is near zero (perfect collinearity)
    if torch.linalg.norm(Ak) < eps:
        # Return zeros if X_k is perfectly explained by X_nk
        # Return zeros with proper shape [d_k] where d_k is feature dimension
        return torch.zeros(X_k.shape[1], device=device)
    
    # Find beta as coefficient between residuals with regularization
    try:
        AtA = torch.t(Ak) @ Ak
        AtA_reg = AtA + lambda_reg * torch.eye(AtA.shape[0], device=device)
        beta = torch.linalg.solve(AtA_reg, torch.t(Ak) @ yk) 
    except:
        # Final fallback
        beta = torch.linalg.pinv(Ak) @ yk
        # beta = torch.linalg.lstsq(Ak, yk, rcond=1e-4, driver='gels')[0]
    
    return beta