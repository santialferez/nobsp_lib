"""
Main NObSP decomposition class.

Provides a unified, scikit-learn compatible interface for neural network
decomposition using oblique subspace projections.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Literal

from .utils.validation import check_is_fitted, validate_data
from .utils.device import auto_detect_device
from .utils.tensor_ops import to_tensor, to_numpy


class NObSP:
    """
    Neural Oblique Subspace Projections for interpretability.
    
    This estimator decomposes neural network predictions into 
    feature-specific contributions using oblique projections.
    
    Parameters
    ----------
    method : {'basic', 'alpha', 'beta'}, default='alpha'
        The decomposition method:
        - 'basic': Oblique projections only (no inference support)
        - 'alpha': Direct coefficient computation 
        - 'beta': Partial regression coefficients
        
    regularization : float, default=1e-4
        Regularization for numerical stability
        
    device : str or torch.device, optional
        Device for computation ('cpu', 'cuda', 'mps', or None for auto)
        
    Attributes
    ----------
    components_ : ndarray
        Learned coefficients (Alpha/Beta) or projections (Basic).
        Shape depends on problem type and method:
        - Regression: (hidden_size, n_features * n_outputs)
        - Classification: (hidden_size, n_features * n_classes)
        - Basic method: (n_samples, n_samples, n_features)
        
    contributions_ : ndarray
        Feature contributions from training data.
        Shape: (n_samples, n_features, n_outputs/n_classes)
        
    n_features_in_ : int
        Number of input features
        
    n_outputs_ : int
        Number of outputs detected
        
    problem_type_ : str
        Detected problem type ('regression' or 'classification')
        
    hidden_size_ : int
        Size of the last hidden layer in the neural network
        
    Examples
    --------
    >>> from nobsp.decomposition import NObSP
    >>> # Assuming you have a trained PyTorch model
    >>> nobsp = NObSP(method='alpha')
    >>> nobsp.fit(X_train, model)
    >>> contributions = nobsp.transform(X_test)
    >>> coefficients = nobsp.components_
    """
    
    def __init__(
        self, 
        method: Literal['basic', 'alpha', 'beta'] = 'alpha',
        regularization: float = 1e-4,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.method = method
        self.regularization = regularization
        self.device = device
        
    def fit(self, X: np.ndarray, model: nn.Module, y=None) -> 'NObSP':
        """
        Compute NObSP decomposition.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        model : torch.nn.Module
            PyTorch model in eval mode. Must accept tensor input and return:
            - For regression: (predictions, activations) 
            - For classification: (probabilities, activations, logits)
            
        y : None
            Ignored. Present for sklearn API compatibility.
            
        Returns
        -------
        self : object
            Fitted estimator
            
        Raises
        ------
        ValueError
            If model output format is not recognized
        """
        # Validate and prepare data
        X = validate_data(X, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        
        # Setup device
        self._device = auto_detect_device() if self.device is None else torch.device(self.device)
        model = model.to(self._device)
        model.eval()
        
        # Get model outputs and detect problem type
        X_tensor = to_tensor(X, self._device)
        with torch.no_grad():
            outputs = model(X_tensor)
            
        self.problem_type_, y_pred = self._validate_model_output(outputs)
        self.n_outputs_ = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
        
        # Get hidden layer size
        self.hidden_size_ = self._get_hidden_size(model)
        
        # Store model reference for basic method
        if self.method == 'basic':
            self._model = model
            
        # Compute decomposition
        self._fit_decomposition(X, y_pred, model)
        
        return self
        
    def transform(self, X: np.ndarray, model: Optional[nn.Module] = None) -> np.ndarray:
        """
        Apply decomposition to get feature contributions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        model : torch.nn.Module, optional
            Model for new data. Required for basic method.
            For alpha/beta methods, uses stored coefficients if None.
            
        Returns
        -------
        contributions : ndarray
            Feature contributions with shape:
            - Regression: (n_samples, n_features) or (n_samples, n_features, n_outputs)
            - Classification: (n_samples, n_features, n_classes)
            
        Raises
        ------
        NotFittedError
            If transform is called before fit
        ValueError
            If basic method is used without providing model
        """
        check_is_fitted(self, ['components_', 'n_features_in_'])
        X = validate_data(X, dtype=np.float32, reset=False)
        
        if self.method == 'basic':
            if model is None:
                model = getattr(self, '_model', None)
                if model is None:
                    raise ValueError(
                        "Basic method requires model for transform. "
                        "Either provide model parameter or use fit_transform()."
                    )
            return self._transform_basic(X, model)
        else:
            # Alpha/Beta methods use stored coefficients
            if model is None:
                # For inference, we need to get feature transformations
                # This requires having the model
                raise ValueError(
                    f"{self.method.capitalize()} method requires model "
                    "to compute feature transformations for new data."
                )
            return self._transform_coefficients(X, model)
            
    def fit_transform(self, X: np.ndarray, model: nn.Module, y=None) -> np.ndarray:
        """
        Fit decomposer and transform data in one step.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        model : torch.nn.Module
            PyTorch model
            
        y : None
            Ignored. Present for sklearn API compatibility.
            
        Returns
        -------
        contributions : ndarray
            Feature contributions
        """
        return self.fit(X, model, y).contributions_
        
    def get_feature_importance(self) -> np.ndarray:
        """
        Calculate global feature importance scores.
        
        Feature importance is computed as the standard deviation
        of centered contributions across samples.
        
        Returns
        -------
        importance : ndarray of shape (n_features, n_outputs)
            Feature importance scores for each output/class
            
        Raises
        ------
        NotFittedError
            If called before fit
        """
        check_is_fitted(self, 'contributions_')
        
        importance = np.zeros((self.n_features_in_, self.n_outputs_))
        
        for i in range(self.n_features_in_):
            for j in range(self.n_outputs_):
                contributions = self.contributions_[:, i, j]
                centered = contributions - contributions.mean()
                importance[i, j] = np.std(centered)
                
        return importance
        
    # Private methods
    
    def _validate_model_output(self, outputs: Union[torch.Tensor, Tuple]) -> Tuple[str, torch.Tensor]:
        """Validate model output format and detect problem type."""
        if isinstance(outputs, tuple):
            if len(outputs) == 3:
                # Classification: (probabilities, activations, logits)
                probabilities, activations, logits = outputs
                # Store all outputs for different methods
                self._probabilities = probabilities
                self._logits = logits
                # Use logits for decomposition (matches old implementation)
                return 'classification', logits
            elif len(outputs) == 2:
                # Regression: (predictions, activations)
                predictions, activations = outputs
                return 'regression', predictions
            else:
                raise ValueError(
                    f"Model returned {len(outputs)} outputs. Expected 2 (regression) or 3 (classification)."
                )
        else:
            raise ValueError(
                "Model must return a tuple. For regression: (predictions, activations). "
                "For classification: (probabilities, activations, logits)."
            )
            
    def _get_hidden_size(self, model: nn.Module) -> int:
        """Extract the size of the last hidden layer."""
        # Try to find the last linear layer
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
                
        if last_linear is None:
            raise ValueError("Could not find any Linear layers in the model")
            
        return last_linear.in_features
        
    def _fit_decomposition(self, X: np.ndarray, y_pred: torch.Tensor, model: nn.Module):
        """Route to appropriate decomposition method."""
        if self.method == 'basic':
            self._fit_basic(X, y_pred, model)
        elif self.method == 'alpha':
            self._fit_alpha(X, y_pred, model)
        elif self.method == 'beta':
            self._fit_beta(X, y_pred, model)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
    def _fit_basic(self, X: np.ndarray, y_pred: torch.Tensor, model: nn.Module):
        """Fit using basic oblique projections."""
        from .core.decompose import decompose_basic
        
        self.components_, self.contributions_ = decompose_basic(
            X, y_pred, model, self.problem_type_, 
            self._device, self.regularization
        )
        
    def _fit_alpha(self, X: np.ndarray, y_pred: torch.Tensor, model: nn.Module):
        """Fit using alpha coefficients."""
        from .core.decompose import decompose_alpha
        
        self.components_, self.contributions_ = decompose_alpha(
            X, y_pred, model, self.problem_type_,
            self._device, self.regularization
        )
        
    def _fit_beta(self, X: np.ndarray, y_pred: torch.Tensor, model: nn.Module):
        """Fit using beta coefficients."""
        from .core.decompose import decompose_beta
        
        self.components_, self.contributions_ = decompose_beta(
            X, y_pred, model, self.problem_type_,
            self._device, self.regularization
        )
        
    def _transform_basic(self, X: np.ndarray, model: nn.Module) -> np.ndarray:
        """Transform using basic method (requires model)."""
        # Basic method needs to recompute projections
        from .core.decompose import apply_basic_transform
        
        return apply_basic_transform(
            X, model, self.components_, self.problem_type_,
            self._device, self.regularization
        )
        
    def _transform_coefficients(self, X: np.ndarray, model: nn.Module) -> np.ndarray:
        """Transform using stored coefficients (alpha or beta)."""
        from .core.decompose import apply_coefficient_transform
        
        return apply_coefficient_transform(
            X, model, self.components_, self.problem_type_,
            self._device, self.n_features_in_, self.n_outputs_
        )