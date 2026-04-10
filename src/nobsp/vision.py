"""
Vision module for NObSP-CAM with scikit-learn compatible API.

This module provides a unified interface for CNN interpretability using
NObSP-CAM, following scikit-learn conventions for ease of use.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Literal
import warnings

from .core.nobsp_cam import NObSPCAM
from .utils.validation import check_is_fitted
from .utils.device import auto_detect_device


class NObSPVision:
    """
    Neural Oblique Subspace Projections for CNN interpretability.
    
    A scikit-learn compatible interface for generating Class Activation Maps
    using NObSP decomposition on CNN features. This provides interpretable
    visualizations of what regions in an image contribute to model predictions.
    
    Parameters
    ----------
    method : {'alpha', 'beta'}, default='beta'
        The decomposition method:
        - 'alpha': Direct coefficient computation
        - 'beta': Partial regression coefficients (recommended)
        
    target_layer : str or None, default=None
        Name/path of target convolutional layer for feature extraction.
        If None, auto-detects the last convolutional layer.
        Examples: 'layer4', 'layer4.1.relu', 'features.29'
        
    device : str or torch.device or None, default=None
        Device for computation ('cpu', 'cuda', 'mps', or None for auto-detect)
        
    regularization : float, default=1e-6
        Regularization parameter for numerical stability in decomposition

    flatten_strategy : {'channel', 'element'}, default='channel'
        Strategy used when the model's classifier consumes flattened spatial
        features (e.g., VGG). 'channel' averages spatial contributions per
        channel (legacy behavior). 'element' retains per-element contributions
        and combines them with activations pixel-wise when generating CAMs.
    decomposition_space : {'hidden', 'classifier_input'}, default='classifier_input'
        Selects which feature space to decompose when the classifier contains
        hidden layers. "hidden" matches earlier releases by decomposing the
        4,096-dimensional hidden representation for VGG. "classifier_input"
        decomposes the flattened spatial tensor fed into the classifier for
        higher-fidelity spatial explanations.
        
    Attributes
    ----------
    coefficients_ : ndarray
        Learned NObSP coefficients after calibration.
        Shape: (hidden_size, n_features * n_classes)
        
    is_fitted_ : bool
        Whether the model has been calibrated with fit()
        
    n_classes_ : int
        Number of classifier outputs inferred at calibration time
        
    Examples
    --------
    >>> import torchvision.models as models
    >>> from torch.utils.data import DataLoader
    >>> from nobsp.vision import NObSPVision
    >>> 
    >>> # Load pre-trained model
    >>> model = models.resnet18(pretrained=True)
    >>> 
    >>> # Initialize NObSP-CAM
    >>> nobsp_vision = NObSPVision(method='beta', target_layer='layer4')
    >>> 
    >>> # Calibrate on dataset
    >>> nobsp_vision.fit(calibration_loader, model, max_samples=500)
    >>> 
    >>> # Generate CAMs for new images
    >>> cams = nobsp_vision.transform(test_images, model)
    >>> 
    >>> # Access heatmaps and contributions
    >>> heatmap = cams[0]['heatmap']
    >>> contributions = cams[0]['contributions']
    """
    
    def __init__(
        self,
        method: str = 'beta',
        target_layer: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        regularization: float = 1e-6,
        flatten_strategy: str = 'channel',
        decomposition_space: Literal["hidden", "classifier_input"] = "classifier_input"
    ):
        # Validate parameters
        if method not in ['alpha', 'beta']:
            raise ValueError(f"method must be 'alpha' or 'beta', got {method}")
        if flatten_strategy not in {'channel', 'element'}:
            raise ValueError(
                f"flatten_strategy must be 'channel' or 'element', got {flatten_strategy}"
            )

        self.method = method
        self.target_layer = target_layer
        self.device = device
        self.regularization = regularization
        self.flatten_strategy = flatten_strategy
        if decomposition_space not in {"hidden", "classifier_input"}:
            raise ValueError(
                "decomposition_space must be 'hidden' or 'classifier_input', "
                f"got {decomposition_space}"
            )
        self.decomposition_space = decomposition_space
        
        # Will be set during fit
        self._nobsp_cam = None
        self.coefficients_ = None
        self.is_fitted_ = False
        self.n_classes_ = None
        self.calibration_targets_: Optional[np.ndarray] = None
        self.selected_classes_: Optional[List[int]] = None
        
    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        **calibration_kwargs: Any,
    ) -> 'NObSPVision':
        """
        Calibrate NObSP-CAM by computing coefficients on a dataset.
        
        This calibration step is essential for generating meaningful CAMs.
        NObSP requires multiple samples to compute proper oblique projections.
        
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader with calibration images. Should yield (images, labels) tuples.
            
        model : torch.nn.Module
            Pre-trained CNN model (e.g., ResNet, VGG). Must be in eval mode
            or will be set to eval mode automatically.
            
        max_samples : int, optional
            Maximum number of samples to use for calibration. If None, uses all
            samples in the dataloader. Using 500-1000 samples is typically sufficient.
            
        verbose : bool, default=True
            Whether to print progress information during calibration.

        **calibration_kwargs :
            Additional keyword arguments forwarded to ``NObSPCAM.calibrate``.
            
        Returns
        -------
        self : NObSPVision
            Fitted estimator
            
        Notes
        -----
        The calibration process:
        1. Extracts features from the target convolutional layer
        2. Pools features to get channel-wise representations
        3. Computes NObSP coefficients using the specified method
        4. Caches coefficients for efficient CAM generation
        """
        # Set model to eval mode
        model.eval()
        
        # Create internal NObSPCAM instance
        self._nobsp_cam = NObSPCAM(
            model=model,
            target_layer=self.target_layer,
            method=self.method,
            device=self.device,
            regularization=self.regularization,
            flatten_strategy=self.flatten_strategy,
            decomposition_space=self.decomposition_space
        )
        
        # Perform calibration
        self._nobsp_cam.calibrate(
            calibration_loader=dataloader,
            max_samples=max_samples,
            verbose=verbose,
            **calibration_kwargs,
        )

        # Store attributes for sklearn compatibility
        self.coefficients_ = self._nobsp_cam.cached_coefficients
        self.is_fitted_ = True
        cached_selected = getattr(self._nobsp_cam, "cached_selected_classes", None)
        if cached_selected is not None:
            self.selected_classes_ = list(cached_selected)
            self.n_classes_ = len(self.selected_classes_)
        else:
            self.selected_classes_ = None
            self.n_classes_ = self._nobsp_cam.fc_wrapper.out_features
        targets = getattr(self._nobsp_cam, "last_calibration_targets", None)
        if targets is not None:
            self.calibration_targets_ = targets.numpy()
        else:
            self.calibration_targets_ = None

        return self
        
    def transform(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        model: Optional[nn.Module] = None,
        target_classes: Optional[Union[int, List[int]]] = None,
        return_features: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate NObSP-CAMs for input images.
        
        Parameters
        ----------
        images : torch.Tensor or list of torch.Tensor
            Input images to generate CAMs for. Should be pre-processed tensors
            of shape [B, 3, H, W] or list of individual image tensors.
            
        model : torch.nn.Module, optional
            CNN model to use. If None, uses the model from fit().
            Must be the same architecture as the model used in fit().
            
        target_classes : int or list of int, optional
            Target class(es) for explanation. Can be:
            - None: uses predicted class for each image
            - int: same target class for all images
            - list: specific target class for each image
            
        return_features : bool, default=False
            Whether to return intermediate tensors (spatial maps, flattened
            classifier input, decomposition feature vector, logits) in the
            results dictionary.
            
        Returns
        -------
        results : list of dict
            List of CAM results for each image. Each dictionary contains:
            - 'heatmap': Normalized CAM [H, W] upsampled to input size
            - 'heatmap_positive': CAM from positive contributions only
            - 'heatmap_negative': CAM from negative contributions only  
            - 'contributions': Channel contributions for target class [C,]
            - 'predicted_class': Model's predicted class index
            - 'target_class': Class used for CAM generation
            - 'features_shape': Shape of spatial features (if return_features=True)
            
        Raises
        ------
        NotFittedError
            If transform is called before fit
        ValueError
            If images have incorrect shape or model architecture mismatch
            
        Examples
        --------
        >>> # Generate CAMs for predicted classes
        >>> cams = nobsp_vision.transform(test_images, model)
        >>> 
        >>> # Generate CAMs for specific class
        >>> cams = nobsp_vision.transform(test_images, model, target_classes=217)
        >>> 
        >>> # Visualize the heatmap
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(cams[0]['heatmap'], cmap='jet')
        >>> plt.colorbar()
        >>> plt.show()
        """
        # Check if fitted
        if not self.is_fitted_:
            raise ValueError(
                "This NObSPVision instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using transform."
            )
            
        if self._nobsp_cam is None:
            raise ValueError("Internal NObSPCAM not initialized. Please call fit() first.")
            
        # Handle single image
        if isinstance(images, torch.Tensor):
            if len(images.shape) == 3:  # Single image [3, H, W]
                images = images.unsqueeze(0)
            batch_input = True
        else:
            # List of images
            images = torch.stack(images) if isinstance(images, list) else images
            batch_input = False
            
        # Handle target_classes
        if target_classes is not None:
            if isinstance(target_classes, int):
                target_classes = [target_classes] * len(images)
            elif len(target_classes) != len(images):
                raise ValueError(
                    f"target_classes length ({len(target_classes)}) must match "
                    f"number of images ({len(images)})"
                )
        else:
            target_classes = [None] * len(images)
            
        # Generate CAMs for each image
        results = []
        for img, target_class in zip(images, target_classes):
            if len(img.shape) == 3:
                img = img.unsqueeze(0)  # Add batch dimension
                
            cam_result = self._nobsp_cam(
                image=img,
                target_class=target_class,
                return_features=return_features
            )
            results.append(cam_result)
            
        return results
        
    def fit_transform(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        images: Union[torch.Tensor, List[torch.Tensor]],
        target_classes: Optional[Union[int, List[int]]] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fit NObSP-CAM and transform images in one step.
        
        Convenience method that combines fit() and transform().
        
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for calibration
            
        model : torch.nn.Module
            Pre-trained CNN model
            
        images : torch.Tensor or list of torch.Tensor
            Images to generate CAMs for after calibration
            
        target_classes : int or list of int, optional
            Target classes for CAM generation
            
        max_samples : int, optional
            Maximum calibration samples
            
        verbose : bool, default=True
            Whether to print progress
            
        Returns
        -------
        results : list of dict
            CAM results for each image
        """
        return self.fit(dataloader, model, max_samples, verbose).transform(
            images, model, target_classes
        )
        
    def save_model(self, filepath: Union[str, Path], metadata: Optional[Dict] = None):
        """
        Save calibrated NObSP-CAM coefficients and configuration.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the model (will add .npz extension if not present)
            
        metadata : dict, optional
            Additional metadata to save with the model
            
        Raises
        ------
        ValueError
            If model hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
            
        if self._nobsp_cam is None:
            raise ValueError("Internal NObSPCAM not initialized.")
            
        # Prepare metadata
        save_metadata = {
            'method': self.method,
            'target_layer': self.target_layer,
            'regularization': self.regularization,
            'num_classes': self.n_classes_,
            'flatten_strategy': self.flatten_strategy,
            'decomposition_space': self.decomposition_space,
        }
        if metadata:
            save_metadata.update(metadata)
        if self.selected_classes_ is not None:
            save_metadata['selected_classes'] = np.asarray(
                self.selected_classes_, dtype=np.int64
            )
            
        # Use internal save method
        self._nobsp_cam.save_coefficients_with_metadata(str(filepath), save_metadata)
        
    def load_model(self, filepath: Union[str, Path], model: nn.Module) -> Dict:
        """
        Load previously calibrated NObSP-CAM coefficients.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved model file
            
        model : torch.nn.Module
            CNN model with same architecture as used during calibration
            
        Returns
        -------
        metadata : dict
            Dictionary containing saved metadata and configuration
            
        Notes
        -----
        After loading, the model is ready to use with transform() without
        needing to call fit() again.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        # Load the saved data
        with np.load(str(filepath), allow_pickle=True) as data:
            if 'method' in data:
                self.method = str(data['method'])
            if 'regularization' in data:
                self.regularization = float(data['regularization'])
            if 'target_layer' in data:
                self.target_layer = (
                    str(data['target_layer']) if data['target_layer'] is not None else None
                )
            if 'flatten_strategy' in data:
                loaded_strategy = str(data['flatten_strategy'])
                if loaded_strategy in {'channel', 'element'}:
                    self.flatten_strategy = loaded_strategy
                else:
                    warnings.warn(
                        f"Unknown flatten_strategy '{loaded_strategy}' in saved model; keeping current value '{self.flatten_strategy}'."
                    )
            if 'decomposition_space' in data:
                loaded_space = str(data['decomposition_space'])
                if loaded_space in {"hidden", "classifier_input"}:
                    self.decomposition_space = loaded_space
                else:
                    warnings.warn(
                        f"Unknown decomposition_space '{loaded_space}' in saved model; keeping current value '{self.decomposition_space}'."
                    )

        # Create internal NObSPCAM with loaded configuration
        self._nobsp_cam = NObSPCAM(
            model=model,
            target_layer=self.target_layer,
            method=self.method,
            device=self.device,
            regularization=self.regularization,
            flatten_strategy=self.flatten_strategy,
            decomposition_space=self.decomposition_space
        )
        
        # Load coefficients
        metadata = self._nobsp_cam.load_coefficients_with_metadata(str(filepath))
        
        # Update attributes
        self.coefficients_ = self._nobsp_cam.cached_coefficients
        self.is_fitted_ = True
        cached_selected = getattr(self._nobsp_cam, "cached_selected_classes", None)
        if cached_selected is not None:
            self.selected_classes_ = list(cached_selected)
            self.n_classes_ = len(self.selected_classes_)
        else:
            self.selected_classes_ = None
            self.n_classes_ = self._nobsp_cam.fc_wrapper.out_features
        
        return metadata
        
    def get_feature_importance(self) -> np.ndarray:
        """
        Calculate global channel importance scores.
        
        Channel importance is computed as the mean absolute value of
        coefficients across all classes.
        
        Returns
        -------
        importance : ndarray of shape (n_channels,)
            Channel importance scores
            
        Raises
        ------
        NotFittedError
            If called before fit
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if self.coefficients_ is None:
            raise ValueError("No coefficients available.")
            
        # Coefficients shape: (hidden_size, n_features * n_classes)
        # For CNNs, hidden_size = n_channels (after pooling)
        n_channels = self._nobsp_cam.fc_wrapper.in_features
        n_classes = self.n_classes_
        
        # The coefficients are already in the right shape
        # Each row corresponds to a channel, columns are classes
        # So we just need to compute mean absolute value across classes
        if self.coefficients_.shape[0] == n_channels:
            # Reshape if needed to separate classes
            if self.coefficients_.shape[1] == n_channels * n_classes:
                # Beta method stores flattened coefficients
                coeffs_reshaped = self.coefficients_.reshape(n_channels, -1)
                importance = np.abs(coeffs_reshaped).mean(axis=1)
            else:
                # Already in correct shape
                importance = np.abs(self.coefficients_).mean(axis=1)
        else:
            # Fallback: just return mean across all dimensions
            importance = np.abs(self.coefficients_).mean(axis=1)
        
        return importance[:n_channels]  # Ensure we return correct number of channels
        
    def __repr__(self):
        """String representation of the estimator."""
        class_name = self.__class__.__name__
        params = []
        params.append(f"method='{self.method}'")
        if self.target_layer:
            params.append(f"target_layer='{self.target_layer}'")
        params.append(f"regularization={self.regularization}")
        params.append(f"flatten_strategy='{self.flatten_strategy}'")
        params.append(f"decomposition_space='{self.decomposition_space}'")
        
        fitted_status = " (fitted)" if self.is_fitted_ else " (not fitted)"
        return f"{class_name}({', '.join(params)}){fitted_status}"
