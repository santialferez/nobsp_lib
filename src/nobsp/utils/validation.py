"""
Validation utilities for NObSP.

Provides input validation and sklearn-compatible checks.
"""

import numpy as np
from typing import Optional


class NotFittedError(ValueError, AttributeError):
    """Exception raised when transform is called before fit."""
    pass


def check_is_fitted(estimator, attributes=None):
    """
    Check if estimator is fitted by verifying the presence of fitted attributes.
    
    Parameters
    ----------
    estimator : estimator instance
        Estimator to check
    attributes : str or list of str, optional
        Attributes to check. If None, checks for any attribute ending with '_'
        
    Raises
    ------
    NotFittedError
        If the estimator is not fitted
    """
    if attributes is None:
        # Check for any fitted attributes (ending with _)
        fitted_attrs = [attr for attr in dir(estimator) if attr.endswith('_') and not attr.startswith('_')]
        if not fitted_attrs:
            raise NotFittedError(f"This {type(estimator).__name__} instance is not fitted yet.")
    else:
        # Check specific attributes
        if isinstance(attributes, str):
            attributes = [attributes]
        
        missing = []
        for attr in attributes:
            if not hasattr(estimator, attr):
                missing.append(attr)
                
        if missing:
            raise NotFittedError(
                f"This {type(estimator).__name__} instance is not fitted yet. "
                f"Missing attributes: {missing}"
            )


def validate_data(
    X, 
    dtype=None, 
    ensure_2d=True, 
    allow_nan=False,
    reset=True
) -> np.ndarray:
    """
    Validate input data.
    
    Parameters
    ----------
    X : array-like
        Input data
    dtype : dtype, optional
        Desired data type
    ensure_2d : bool, default=True
        Whether to ensure X is 2D
    allow_nan : bool, default=False
        Whether to allow NaN values
    reset : bool, default=True
        Whether this is initial validation (vs transform validation)
        
    Returns
    -------
    X_validated : np.ndarray
        Validated array
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Convert to numpy array
    X = np.asarray(X)
    
    # Check dtype
    if dtype is not None:
        X = X.astype(dtype, copy=False)
    
    # Ensure 2D
    if ensure_2d and X.ndim == 1:
        X = X.reshape(-1, 1)
    elif ensure_2d and X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
    
    # Check for NaN
    if not allow_nan and np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")
    
    return X


def validate_model_output(outputs):
    """
    Validate model output format and return problem type.
    
    Parameters
    ----------
    outputs : tuple or tensor
        Model outputs
        
    Returns
    -------
    problem_type : str
        'regression' or 'classification'
    predictions : tensor
        The predictions from the model
        
    Raises
    ------
    ValueError
        If output format is not recognized
    """
    if not isinstance(outputs, tuple):
        raise ValueError(
            "Model must return a tuple. Expected formats:\n"
            "- Regression: (predictions, activations)\n"
            "- Classification: (probabilities, activations, logits)"
        )
    
    if len(outputs) == 2:
        # Regression format
        predictions, activations = outputs
        return 'regression', predictions
    elif len(outputs) == 3:
        # Classification format
        probabilities, activations, logits = outputs
        return 'classification', probabilities
    else:
        raise ValueError(
            f"Model returned {len(outputs)} outputs. Expected 2 (regression) or 3 (classification)."
        )