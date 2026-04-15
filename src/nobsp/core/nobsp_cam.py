#!/usr/bin/env python3
"""
NObSP-CAM: Neural Oblique Subspace Projections for CNN Interpretability

This module implements NObSP-CAM for generating class activation maps using
NObSP decomposition on CNN features. It directly uses the CNN's FC layer
for decomposition, avoiding unnecessary surrogate classifiers.
"""

import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Tuple, Union, Dict, List, Literal, Sequence, Any
import warnings

# Import NObSP decomposition functions from new API
from nobsp.core.decompose import decompose_beta, decompose_alpha
from nobsp.core.beta_batched import beta_calibrate_batched
from nobsp.utils.tensor_ops import to_tensor, to_numpy


class FCWrapper(nn.Module):
    """
    Wrapper to make CNN's FC layer compatible with NObSP API.
    
    NObSP expects models to return (prob, X_trans, y_lin) for classification.
    This wrapper adapts a standard FC layer to this interface.
    
    Parameters
    ----------
    fc_layer : nn.Module
        The FC layer from a pre-trained CNN
    """
    
    def __init__(self, fc_layer: nn.Module):
        super().__init__()
        self.fc = fc_layer
        # Store dimensions for NObSP
        self.in_features = fc_layer.in_features
        self.out_features = fc_layer.out_features
        
    def forward(self, x):
        """
        Forward pass returning format required by NObSP API.
        
        Returns
        -------
        prob : torch.Tensor
            Softmax probabilities [B, num_classes]
        X_trans : torch.Tensor
            Input features (used as transformation) [B, in_features]
        y_lin : torch.Tensor
            Linear output (logits) [B, num_classes]
        """
        y_lin = self.fc(x)
        prob = F.softmax(y_lin, dim=1)
        # For FC layer, X itself serves as the transformation
        return prob, x, y_lin
    
    def children(self):
        """Return FC layer as child for NObSP compatibility."""
        return [self.fc]


class FullClassifierWrapper(nn.Module):
    """Wrapper that runs an entire classifier stack on flattened features."""

    def __init__(
        self,
        classifier: nn.Module,
    ):
        super().__init__()
        if not isinstance(classifier, nn.Sequential):
            raise TypeError(
                "FullClassifierWrapper expects an nn.Sequential classifier (e.g., VGG head)"
            )

        # Clone the classifier to keep a stable copy without reparenting modules
        self.classifier = copy.deepcopy(classifier)
        self.classifier.eval()

        modules = list(self.classifier.children())
        if not modules:
            raise ValueError("Classifier sequential is empty")

        if not isinstance(modules[-1], nn.Linear):
            raise ValueError(
                "Last module in classifier must be nn.Linear for NObSP decomposition"
            )

        self.final_linear = modules[-1]
        self.prefix = nn.Sequential(*modules[:-1]) if len(modules) > 1 else nn.Identity()

        self.out_features = self.final_linear.out_features
        self.in_features = self._infer_input_dim()

    def _infer_input_dim(self) -> int:
        if isinstance(self.prefix, nn.Identity):
            return self.final_linear.in_features
        for module in self.prefix.children():
            in_feat = getattr(module, 'in_features', None)
            if in_feat is not None:
                return in_feat
        return self.final_linear.in_features

    def forward(self, x: torch.Tensor):
        hidden = self.prefix(x)
        logits_full = self.final_linear(hidden)
        prob = F.softmax(logits_full, dim=1)
        return prob, hidden, logits_full

    def map_to_reduced(self, full_idx: int) -> Optional[int]:
        if self.full_to_reduced is None:
            return None
        return self.full_to_reduced.get(full_idx)

    def map_to_full(self, reduced_idx: int) -> int:
        if self.reduced_to_full is None:
            return reduced_idx
        return self.reduced_to_full.get(reduced_idx, -1)

class NObSPCAM:
    """
    NObSP-CAM for CNN interpretability using direct FC layer decomposition.
    
    This implementation uses the CNN's own FC layer for NObSP decomposition,
    treating pooled CNN features as input features for oblique projection.
    
    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained CNN model (e.g., ResNet, VGG)
    target_layer : str or None
        Name/path of target convolutional layer for feature extraction
    method : str
        Decomposition method: 'alpha' or 'beta' (default: 'beta')
    device : str or torch.device, optional
        Device for computation (auto-detected if None)
    regularization : float
        Regularization parameter for NObSP decomposition
    flatten_strategy : {'channel', 'element'}, default='channel'
        How to handle contributions when the classifier consumes flattened
        convolutional features (e.g., VGG). The default 'channel' behavior
        matches previous releases by averaging spatial contributions per
        channel. The 'element' option retains the per-element contributions
        and combines them with activations pixel-by-pixel to form the CAM.
    decomposition_space : {'hidden', 'classifier_input'}, default='classifier_input'
        Controls which feature space NObSP decomposes when the classifier
        includes hidden layers (e.g., VGG). "hidden" reproduces the previous
        behavior by decomposing the hidden 4,096-dimensional layer. "classifier_input"
        computes contributions directly on the flattened spatial tensor passed
        into the classifier stack.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        method: str = 'beta',
        device: Optional[Union[str, torch.device]] = None,
        regularization: float = 1e-6,
        flatten_strategy: str = 'channel',
        decomposition_space: Literal["hidden", "classifier_input"] = "classifier_input"
    ):
        self.model = model.eval()
        self.method = method
        self.regularization = regularization
        if flatten_strategy not in {"channel", "element"}:
            raise ValueError(
                f"flatten_strategy must be 'channel' or 'element', got {flatten_strategy}"
            )
        self.flatten_strategy = flatten_strategy
        if decomposition_space not in {"hidden", "classifier_input"}:
            raise ValueError(
                "decomposition_space must be 'hidden' or 'classifier_input', "
                f"got {decomposition_space}"
            )
        self.decomposition_space = decomposition_space
        self._uses_classifier_input = self.decomposition_space == "classifier_input"

        # Auto-detect device
        if device is None:
            self.device = (torch.device('cuda') if torch.cuda.is_available()
                          else torch.device('mps') if torch.backends.mps.is_available()
                          else torch.device('cpu'))
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
            
        self.model = self.model.to(self.device)

        # Identify target convolutional layer
        self.target_layer = self._identify_target_layer(target_layer)

        # Prepare classifier prefix modules (layers before final linear)
        self.classifier_prefix_modules: List[nn.Module] = []

        # Track avgpool if available for projecting spatial features
        self.avgpool_layer = getattr(self.model, "avgpool", None)

        # Cache classifier module if present (e.g., VGG)
        self.classifier_module = getattr(self.model, "classifier", None)

        # Get FC layer and wrap it for NObSP
        self.fc_layer = self._get_fc_layer()

        can_use_full_classifier = (
            self._uses_classifier_input
            and isinstance(self.classifier_module, nn.Sequential)
            and len(list(self.classifier_module.children())) > 0
        )

        if can_use_full_classifier:
            self.fc_wrapper = FullClassifierWrapper(
                classifier=self.classifier_module,
            ).to(self.device)
        else:
            # Use full FC wrapper for all classes
            self.fc_wrapper = FCWrapper(self.fc_layer).to(self.device)

        self.fc_wrapper.eval()
        self._full_classifier_available = can_use_full_classifier
        self._last_classifier_input_shape: Optional[Tuple[int, int, int]] = None

        # Storage for features from hooks
        self.features_spatial = None
        self.hook_handle = None

        # Cache for coefficients (computed once, reused for efficiency)
        self.cached_coefficients = None
        self.cached_selected_classes: Optional[List[int]] = None
        self.cached_beta_backend: Optional[str] = None
        self.cached_checkpoint_path: Optional[str] = None
        self.last_calibration_metadata: Dict[str, Any] = {}

        # Track the labels observed during the most recent calibration run
        self.last_calibration_targets: Optional[torch.Tensor] = None

        # Reference first linear module in prefix for backprojection (if any)
        self.backproject_linear: Optional[nn.Linear] = None
        for module in self.classifier_prefix_modules:
            if isinstance(module, nn.Linear):
                self.backproject_linear = module
                break
        
    def _identify_target_layer(self, target_layer_name: Optional[str]) -> nn.Module:
        """Identify the target convolutional layer for feature extraction.
        
        Parameters
        ----------
        target_layer_name : str or None
            Explicit layer name to hook. If None, auto-detects.
        """
        if target_layer_name:
            # User specified layer
            parts = target_layer_name.split('.')
            layer = self.model
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            return layer
        
        # Auto-detect last conv layer based on model architecture
        if hasattr(self.model, 'layer4'):  # ResNet
            # Hook the final ReLU in the last block
            return self.model.layer4[-1].relu
        elif hasattr(self.model, 'features'):  # VGG, AlexNet
            # Find last conv layer in features
            for module in reversed(list(self.model.features.modules())):
                if isinstance(module, nn.Conv2d):
                    return module
        elif hasattr(self.model, 'conv_head'):  # EfficientNet
            return self.model.conv_head
        else:
            raise ValueError("Cannot auto-detect target layer. Please specify target_layer parameter.")
    
    def _get_fc_layer(self) -> nn.Module:
        """Get the FC layer from the model."""
        if hasattr(self.model, 'fc'):
            self.classifier_prefix_modules = []
            return self.model.fc
        elif hasattr(self.model, 'classifier'):
            # Some models have classifier as Sequential, get last Linear layer
            classifier = self.model.classifier
            if isinstance(classifier, nn.Sequential):
                modules = list(classifier.children())
                if modules:
                    *prefix, last = modules
                    self.classifier_prefix_modules = prefix
                    if isinstance(last, nn.Linear):
                        return last
                    # Fallback: if last layer not linear, search reversed
                    for module in reversed(modules):
                        if isinstance(module, nn.Linear):
                            return module
                return classifier
            else:
                self.classifier_prefix_modules = []
                return classifier
        else:
            raise ValueError("Cannot find FC layer in model")

    def _hook_features(self, module, input, output):
        """Hook to capture feature maps from target layer."""
        # Simply capture the output features
        self.features_spatial = output.detach().clone()
    
    def extract_features(
        self,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return spatial maps, flattened classifier input, selected features, logits."""
        # Register forward hook to capture features
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_features)
        
        # Forward pass through full model
        with torch.no_grad():
            logits_full = self.model(image.to(self.device))  # Trigger hook and capture logits

        # Remove hook
        self.hook_handle.remove()

        # Get spatial features from hook
        features_spatial = self.features_spatial
        self.features_spatial = None  # Clear storage

        # Compute classifier input (flattened spatial tensor) and hidden features
        classifier_input, hidden_features = self._project_to_fc_input(features_spatial)

        feature_vector = classifier_input if self._uses_classifier_input else hidden_features

        # Get logits from fc_wrapper (reduced if in reduced mode)
        with torch.no_grad():
            _, _, logits = self.fc_wrapper(feature_vector)

        return features_spatial, classifier_input, feature_vector, logits

    def _project_to_fc_input(
        self, features_spatial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both flattened classifier input and hidden representation."""

        pooled = features_spatial
        if self.avgpool_layer is not None:
            pooled = self.avgpool_layer(pooled)

        # Track shape for reshaping contributions later on
        self._last_classifier_input_shape = (
            pooled.shape[1],
            pooled.shape[2] if pooled.dim() > 2 else 1,
            pooled.shape[3] if pooled.dim() > 2 else 1,
        )

        classifier_input = torch.flatten(pooled, 1)

        hidden = classifier_input
        if self.classifier_prefix_modules:
            hidden = classifier_input
            for module in self.classifier_prefix_modules:
                hidden = module(hidden)

        return classifier_input, hidden

    def save_coefficients(self, filepath: str):
        """Save calibration coefficients to disk."""
        if self.cached_coefficients is None:
            raise ValueError("No coefficients to save. Run calibrate() first.")
        np.save(filepath, self.cached_coefficients)
        print(f"Saved coefficients to {filepath}")
    
    def load_coefficients(self, filepath: str):
        """Load calibration coefficients from disk."""
        self.cached_coefficients = np.load(filepath)
        self.cached_selected_classes = None
        self.cached_beta_backend = None
        self.cached_checkpoint_path = None
        self.last_calibration_metadata = {}
        print(f"Loaded coefficients from {filepath} (shape: {self.cached_coefficients.shape})")
    
    def save_coefficients_with_metadata(self, filepath: str, metadata: Optional[Dict] = None):
        """
        Save calibration coefficients with metadata to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the coefficients (will use .npz extension)
        metadata : dict, optional
            Additional metadata to save with coefficients
        """
        if self.cached_coefficients is None:
            raise ValueError("No coefficients to save. Run calibrate() first.")
        
        # Prepare save data
        save_data = {
            'coefficients': self.cached_coefficients,
            'method': self.method,
            'regularization': self.regularization,
            'flatten_strategy': self.flatten_strategy,
            'decomposition_space': self.decomposition_space,
            'num_classes': getattr(self.fc_wrapper, 'out_features', None),
        }

        if self.cached_beta_backend is not None:
            save_data['beta_backend'] = self.cached_beta_backend
        if self.cached_selected_classes is not None:
            save_data['selected_classes'] = np.asarray(
                self.cached_selected_classes, dtype=np.int64
            )
        if self.cached_checkpoint_path is not None:
            save_data['checkpoint_path'] = self.cached_checkpoint_path
        if self.last_calibration_metadata:
            save_data['calibration_metadata'] = self.last_calibration_metadata
        
        # Add user-provided metadata
        if metadata:
            save_data.update(metadata)
        
        # Save as compressed numpy archive
        np.savez_compressed(filepath, **save_data)
        print(f"Saved coefficients with metadata to {filepath}")

    def load_coefficients_with_metadata(self, filepath: str) -> Dict:
        """
        Load calibration coefficients with metadata from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved coefficients file
            
        Returns
        -------
        metadata : dict
            Dictionary containing all saved metadata
        """
        # Load the archive
        data = np.load(filepath, allow_pickle=True)
        
        # Extract coefficients
        self.cached_coefficients = data['coefficients']
        self.cached_selected_classes = None
        self.cached_beta_backend = None
        self.cached_checkpoint_path = None
        self.last_calibration_metadata = {}

        if 'is_reduced' in data.files:
            raise ValueError(
                "Loaded coefficients were generated with an unsupported legacy reduction. "
                "Please recalibrate with the current NObSP-CAM API."
            )

        if 'selected_classes' in data.files:
            selected_arr = data['selected_classes']
            if selected_arr is not None:
                selected_np = np.asarray(selected_arr)
                if selected_np.size > 0:
                    self.cached_selected_classes = [
                        int(x) for x in selected_np.reshape(-1).tolist()
                    ]

        if 'beta_backend' in data.files:
            backend_field = data['beta_backend']
            if isinstance(backend_field, np.ndarray):
                self.cached_beta_backend = str(backend_field.reshape(-1)[0])
            else:
                self.cached_beta_backend = str(backend_field)

        if 'checkpoint_path' in data.files:
            checkpoint_field = data['checkpoint_path']
            if isinstance(checkpoint_field, np.ndarray):
                checkpoint_value = checkpoint_field.reshape(-1)[0]
                if checkpoint_value is not None:
                    self.cached_checkpoint_path = str(checkpoint_value)
            elif checkpoint_field is not None:
                self.cached_checkpoint_path = str(checkpoint_field)

        if 'calibration_metadata' in data.files:
            meta_field = data['calibration_metadata']
            if isinstance(meta_field, np.ndarray) and meta_field.dtype == object:
                meta_item = meta_field.reshape(-1)[0]
                if isinstance(meta_item, dict):
                    self.last_calibration_metadata = dict(meta_item)
            elif isinstance(meta_field, dict):
                self.last_calibration_metadata = dict(meta_field)

        if 'decomposition_space' in data:
            saved_space = str(data['decomposition_space'])
            if saved_space != self.decomposition_space:
                warnings.warn(
                    f"Loaded coefficients computed in decomposition_space='{saved_space}', "
                    f"but current instance is configured for '{self.decomposition_space}'."
                )
        
        metadata = {key: data[key] for key in data.files}
        print(f"Loaded coefficients with metadata from {filepath}")
        print(f"  Coefficients shape: {self.cached_coefficients.shape}")
        if 'num_classes' in metadata:
            print(f"  Classes: {metadata['num_classes']}")
        
        return metadata
    
    def calibrate(
        self,
        calibration_loader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        *,
        beta_backend: str = "legacy",
        feature_batch_size: int = 16,
        class_batch_size: Optional[int] = None,
        solver: str = "chol",
        mixed_precision: Union[str, bool] = True,
        coefficients_only: bool = True,
        selected_classes: Optional[Sequence[int]] = None,
        checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
        checkpoint_interval_features: int = 16,
        checkpoint_interval_minutes: int = 5,
        resume: bool = True,
        overwrite: bool = False,
    ) -> None:
        """
        Calibrate NObSP-CAM by computing coefficients on a dataset.
        
        This is ESSENTIAL for meaningful CAM generation. NObSP requires
        multiple samples to compute proper oblique projections.
        
        Parameters
        ----------
        calibration_loader : DataLoader
            DataLoader with calibration images
        max_samples : int, optional
            Maximum number of samples to use (None = use all)
        verbose : bool
            Print progress information
        """
        backend_requested = (beta_backend or "legacy").lower()
        if backend_requested == "auto":
            backend_requested = (
                "gpu_batched_multiclass" if self.device.type == "cuda" else "legacy"
            )
        if backend_requested not in {"legacy", "gpu_batched_multiclass"}:
            raise ValueError(
                f"Unsupported beta_backend '{beta_backend}'. "
                "Expected 'legacy', 'gpu_batched_multiclass', or 'auto'."
            )

        def _normalize_selected_classes(
            classes: Optional[Sequence[int]],
        ) -> Optional[List[int]]:
            if classes is None:
                return None
            if isinstance(classes, torch.Tensor):
                return [int(x) for x in classes.detach().cpu().tolist()]
            if isinstance(classes, np.ndarray):
                return [int(x) for x in classes.tolist()]
            if isinstance(classes, Sequence) and not isinstance(
                classes, (str, bytes)
            ):
                return [int(x) for x in classes]
            raise TypeError(
                "selected_classes must be None, a tensor, numpy array, or sequence of integers."
            )

        def _resolve_mixed_precision_flag(value: Union[str, bool]) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return True
            value_str = str(value).strip().lower()
            if value_str in {"0", "false", "off", "none", "no"}:
                return False
            return True

        selected_classes_list = _normalize_selected_classes(selected_classes)
        self.cached_coefficients = None
        self.cached_selected_classes = None
        self.cached_beta_backend = None
        self.cached_checkpoint_path = None
        self.last_calibration_metadata = {}

        if verbose:
            print("Calibrating NObSP-CAM...")
            
        all_features = []
        all_logits = []
        all_targets: List[torch.Tensor] = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_loader):
                if isinstance(batch, dict):
                    images = batch.get("images")
                    if images is None:
                        images = batch.get("image")
                    targets = batch.get("labels")
                    if targets is None:
                        targets = batch.get("label")
                    if images is None:
                        raise ValueError(
                            "Calibration dict batches must include an 'images' or 'image' entry."
                        )
                elif isinstance(batch, (list, tuple)):
                    if len(batch) < 2:
                        raise ValueError(
                            "Calibration dataloader must return tuples of (images, targets)."
                        )
                    images, targets = batch[0], batch[1]
                else:
                    raise TypeError(
                        "Unsupported batch type for calibration dataloader: "
                        f"{type(batch)!r}"
                    )

                if max_samples and len(all_features) * calibration_loader.batch_size >= max_samples:
                    break
                    
                # Extract features
                _, _, feature_vector, logits = self.extract_features(images)
                all_features.append(feature_vector.cpu())
                all_logits.append(logits.cpu())

                if targets is not None:
                    if isinstance(targets, torch.Tensor):
                        all_targets.append(targets.detach().cpu())
                    else:
                        all_targets.append(torch.as_tensor(targets).cpu())
                
                if verbose and batch_idx % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * calibration_loader.batch_size} samples...")
        
        # Concatenate all features and logits
        all_features = torch.cat(all_features, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        if max_samples:
            all_features = all_features[:max_samples]
            all_logits = all_logits[:max_samples]

        if all_targets:
            targets_tensor = torch.cat(all_targets, dim=0)
            if max_samples:
                targets_tensor = targets_tensor[:max_samples]
            self.last_calibration_targets = targets_tensor.clone()
        else:
            self.last_calibration_targets = None

        if verbose:
            print(f"  Computing NObSP coefficients on {all_features.shape[0]} samples...")

        total_classes = all_logits.shape[1]
        if selected_classes_list is not None:
            invalid = [cls for cls in selected_classes_list if cls < 0 or cls >= total_classes]
            if invalid:
                raise ValueError(
                    f"selected_classes contains invalid indices relative to model outputs: {invalid}"
                )

        coefficients_np: np.ndarray
        backend_used = "legacy"

        if self.method == "beta" and backend_requested == "gpu_batched_multiclass":
            if self.device.type != "cuda":
                warnings.warn(
                    "beta_backend='gpu_batched_multiclass' requested but CUDA is not available. "
                    "Falling back to legacy CPU implementation.",
                    RuntimeWarning,
                )
            else:
                backend_used = "gpu_batched_multiclass"
                mixed_precision_flag = _resolve_mixed_precision_flag(mixed_precision)
                calibration_logger = logging.getLogger("nobsp.calibration")

                X_tensor = all_features.to(self.device, dtype=torch.float32)
                Y_tensor = all_logits.to(self.device, dtype=torch.float32)

                coefficients_tensor = beta_calibrate_batched(
                    X=X_tensor,
                    Y=Y_tensor,
                    model=self.fc_wrapper,
                    lambda_reg=self.regularization,
                    device=self.device,
                    feature_batch_size=int(feature_batch_size),
                    class_batch_size=int(class_batch_size) if class_batch_size else None,
                    coefficients_only=coefficients_only,
                    use_mixed_precision=mixed_precision_flag,
                    solver=solver,
                    decomposition_space=self.decomposition_space,
                    selected_classes=selected_classes_list,
                    checkpoint_path=checkpoint_path,
                    checkpoint_interval_features=int(checkpoint_interval_features),
                    checkpoint_interval_minutes=int(checkpoint_interval_minutes),
                    resume=resume,
                    overwrite=overwrite,
                    logger=calibration_logger,
                )

                coefficients_np = coefficients_tensor.cpu().numpy()
                self.cached_selected_classes = (
                    None
                    if selected_classes_list is None
                    else list(selected_classes_list)
                )
                self.cached_checkpoint_path = (
                    str(checkpoint_path) if checkpoint_path is not None else None
                )
                self.cached_beta_backend = backend_used
                self.last_calibration_metadata = {
                    "backend": backend_used,
                    "feature_batch_size": int(feature_batch_size),
                    "class_batch_size": int(class_batch_size)
                    if class_batch_size
                    else None,
                    "solver": solver,
                    "mixed_precision": mixed_precision,
                    "coefficients_only": coefficients_only,
                    "selected_classes": (
                        list(selected_classes_list)
                        if selected_classes_list is not None
                        else None
                    ),
                    "checkpoint_path": self.cached_checkpoint_path,
                    "checkpoint_interval_features": int(checkpoint_interval_features),
                    "checkpoint_interval_minutes": int(checkpoint_interval_minutes),
                    "resume": bool(resume),
                    "overwrite": bool(overwrite),
                    "device": str(self.device),
                    "samples": int(all_features.shape[0]),
                    "features": int(all_features.shape[1]),
                    "classes_total": int(total_classes),
                    "decomposition_space": self.decomposition_space,
                    "regularization": float(self.regularization),
                }
                self.cached_coefficients = coefficients_np

                if verbose:
                    print(
                        f"✓ Calibration complete using GPU batched backend. "
                        f"Coefficients shape: {coefficients_np.shape}"
                    )
                return

        # Fallback to legacy implementation (CPU loops)
        X_np = all_features.numpy()

        if self.method == "beta":
            coefficients_np, _ = decompose_beta(
                X=X_np,
                y_pred=all_logits,
                model=self.fc_wrapper,
                problem_type="classification",
                device=self.device,
                regularization=self.regularization,
            )
        else:
            coefficients_np, _ = decompose_alpha(
                X=X_np,
                y_pred=all_logits,
                model=self.fc_wrapper,
                problem_type="classification",
                device=self.device,
                regularization=self.regularization,
            )

        if self.method == "beta" and selected_classes_list is not None:
            n_features = all_features.shape[1]
            column_indices: List[int] = []
            for class_idx in selected_classes_list:
                start = class_idx * n_features
                column_indices.extend(range(start, start + n_features))
            coefficients_np = coefficients_np[:, column_indices]
            self.cached_selected_classes = list(selected_classes_list)
        else:
            self.cached_selected_classes = None

        self.cached_coefficients = coefficients_np
        self.cached_beta_backend = backend_used
        self.cached_checkpoint_path = None
        self.last_calibration_metadata = {
            "backend": backend_used,
            "selected_classes": (
                list(selected_classes_list)
                if selected_classes_list is not None
                else None
            ),
            "samples": int(all_features.shape[0]),
            "features": int(all_features.shape[1]),
            "classes_total": int(total_classes),
            "decomposition_space": self.decomposition_space,
            "regularization": float(self.regularization),
        }

        if verbose:
            print(f"✓ Calibration complete! Coefficients shape: {coefficients_np.shape}")
    
    def compute_nobsp_contributions(
        self,
        feature_vector: torch.Tensor,
        target_class: Optional[int] = None,
        cache_coefficients: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute channel contributions using NObSP decomposition.
        
        This method treats each channel in the pooled features as a "feature"
        and uses the CNN's classifier head for decomposition.
        
        Parameters
        ----------
        feature_vector : torch.Tensor
            Feature vector used for decomposition. When ``decomposition_space``
            is ``'classifier_input'`` this is the flattened classifier input;
            otherwise it is the hidden layer activations supplied to the
            classifier's final linear layer.
        target_class : int, optional
            Target class for explanation. If None, uses the predicted class.
        cache_coefficients : bool
            Whether to cache coefficients for reuse
            
        Returns
        -------
        contributions : np.ndarray
            Channel contributions for target class [C,]
        coefficients : np.ndarray
            NObSP coefficients for all classes
        """
        # Get predictions from FC layer
        with torch.no_grad():
            y_prob, _, y_lin = self.fc_wrapper(feature_vector)
        
        if target_class is None:
            target_class = y_prob.argmax(1).item()
        computation_target = target_class
        if self.cached_selected_classes is not None:
            try:
                class_position = self.cached_selected_classes.index(computation_target)
            except ValueError as exc:
                raise ValueError(
                    f"Target class {computation_target} was not calibrated. "
                    "Re-run calibration including this class or disable class filtering."
                ) from exc
        else:
            class_position = computation_target
        
        # Check for cached coefficients from calibration
        if self.cached_coefficients is not None:
            coefficients = self.cached_coefficients
            X_np = feature_vector.cpu().numpy()
            n_features = X_np.shape[1]
            
            # Create identity-masked versions of input for each feature
            contributions = np.zeros(n_features)
            
            for i in range(n_features):
                # Create input with only feature i active
                X_single = np.zeros_like(X_np)
                X_single[:, i] = X_np[:, i]
                
                # Get transformation through FC layer
                with torch.no_grad():
                    X_single_tensor = torch.from_numpy(X_single).float().to(self.device)
                    _, X_trans, _ = self.fc_wrapper(X_single_tensor)
                    X_trans = X_trans.cpu().numpy()
                
                # Apply coefficients for target class
                # Coefficients are organized as [hidden, n_outputs * n_features]
                coef_idx = class_position * n_features + i
                contributions[i] = (X_trans @ coefficients[:, coef_idx:coef_idx+1]).item()
            
            channel_contributions = contributions
        else:
            # No calibration done - warn user
            warnings.warn(
                "NObSP-CAM not calibrated! Call .calibrate() with a dataset first. "
                "Computing on single sample will give poor results.",
                UserWarning
            )
            
            # Fallback: compute on single sample (poor results expected)
            X_np = feature_vector.cpu().numpy()
            
            # Use NObSP decomposition
            if self.method == 'beta':
                coefficients, contributions = decompose_beta(
                    X=X_np,
                    y_pred=y_lin.cpu(),
                    model=self.fc_wrapper,
                    problem_type='classification',
                    device=self.device,
                    regularization=self.regularization
                )
            else:  # alpha
                coefficients, contributions = decompose_alpha(
                    X=X_np,
                    y_pred=y_lin.cpu(),
                    model=self.fc_wrapper,
                    problem_type='classification',
                    device=self.device,
                    regularization=self.regularization
                )
            
            # Don't cache single-sample coefficients
            channel_contributions = contributions[0, :, computation_target]
        
        return channel_contributions, coefficients
    
    def generate_cam(
        self,
        features_spatial: torch.Tensor,
        contributions: np.ndarray
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Generate CAM by combining spatial features with channel contributions.
        
        Parameters
        ----------
        features_spatial : torch.Tensor
            Spatial feature maps [B, C, H, W]
        contributions : np.ndarray
            Channel contributions [C,]
            
        Returns
        -------
        dict
            Dictionary containing different CAM visualizations
        """
        B, C, H, W = features_spatial.shape
        
        # Convert contributions to tensor
        contrib_tensor = torch.from_numpy(contributions).float().to(self.device)
        contrib_tensor = self._reduce_contributions(contrib_tensor, features_spatial)

        feature_map = features_spatial[0]
        target_H, target_W = feature_map.shape[1], feature_map.shape[2]

        if contrib_tensor.dim() == 1:
            # Previous channel-wise behavior
            contrib_pos = contrib_tensor.clamp_min(0)
            contrib_neg = contrib_tensor.clamp_max(0).abs()

            features_reshaped = feature_map.permute(1, 2, 0)  # [H, W, C]
            features_flat = features_reshaped.reshape(H * W, C)

            abs_sum = contrib_tensor.abs().sum()
            if abs_sum != 0:
                cam_avg = torch.matmul(features_flat, contrib_tensor.abs()) / abs_sum
            else:
                cam_avg = torch.zeros(H * W, device=self.device)

            if contrib_pos.sum() != 0:
                cam_pos = torch.matmul(features_flat, contrib_pos) / contrib_pos.sum()
            else:
                cam_pos = torch.zeros(H * W, device=self.device)

            if contrib_neg.sum() != 0:
                cam_neg = torch.matmul(features_flat, contrib_neg) / contrib_neg.sum()
            else:
                cam_neg = torch.zeros(H * W, device=self.device)

            cam_avg = cam_avg.view(H, W)
            cam_pos = cam_pos.view(H, W)
            cam_neg = cam_neg.view(H, W)

        elif contrib_tensor.dim() == 3:
            # Element-wise weighting for flattening architectures (e.g., VGG)
            if contrib_tensor.shape != (C, H, W):
                if self.avgpool_layer is None:
                    raise ValueError(
                        f"Expected contributions of shape {(C, H, W)} or pooled grid, got {tuple(contrib_tensor.shape)}"
                    )
                pooled_features = self.avgpool_layer(features_spatial)
                pooled_map = pooled_features[0]
                pooled_H, pooled_W = pooled_map.shape[1], pooled_map.shape[2]
                if contrib_tensor.shape != (C, pooled_H, pooled_W):
                    raise ValueError(
                        f"Contributions shape {tuple(contrib_tensor.shape)} incompatible with pooled grid {(C, pooled_H, pooled_W)}"
                    )
                if (pooled_H, pooled_W) != (H, W):
                    # Broadcast pooled contributions back to the spatial grid used for visualization
                    contrib_tensor = F.interpolate(
                        contrib_tensor.unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

            contrib_pos = contrib_tensor.clamp_min(0)
            contrib_neg = contrib_tensor.clamp_max(0).abs()
            contrib_abs = contrib_tensor.abs()

            cam_pos = (feature_map * contrib_pos).sum(dim=0)
            cam_neg = (feature_map * contrib_neg).sum(dim=0)
            cam_avg = (feature_map * contrib_abs).sum(dim=0)
        else:
            raise ValueError(
                "Unexpected contribution tensor shape after reduction: "
                f"dim={contrib_tensor.dim()}"
            )

        # Normalize helpers
        def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                return (cam - cam_min) / (cam_max - cam_min)
            return torch.zeros_like(cam)

        def normalize_signed(cam: torch.Tensor) -> torch.Tensor:
            max_abs = cam.abs().max()
            if max_abs > 0:
                return cam / max_abs
            return torch.zeros_like(cam)

        cam_pos_norm = normalize_cam(cam_pos)
        cam_neg_norm = normalize_cam(cam_neg)
        cam_signed_norm = normalize_signed(cam_pos - cam_neg)
        cam_combined_norm = normalize_cam(cam_pos_norm - cam_neg_norm)

        return {
            'cam': normalize_cam(cam_avg),
            'cam_positive': cam_pos_norm,
            'cam_negative': cam_neg_norm,
            'cam_signed': cam_signed_norm,
            'cam_combined': cam_combined_norm,
            'contributions': contrib_tensor.detach().cpu().numpy(),
            'contributions_raw': contrib_tensor.cpu().numpy()
        }

    def _reduce_contributions(
        self,
        contrib_tensor: torch.Tensor,
        features_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """Map contributions to channel dimension when classifier uses hidden layers."""
        C = features_spatial.shape[1]

        if self._uses_classifier_input:
            shape = self._last_classifier_input_shape
            if shape is None:
                shape = (C, features_spatial.shape[2], features_spatial.shape[3])
            channels, height, width = shape
            total_elements = channels * height * width

            if contrib_tensor.numel() == total_elements:
                contrib_tensor = contrib_tensor.view(channels, height, width)
                if self.flatten_strategy == 'element':
                    return contrib_tensor
                return contrib_tensor.view(channels, -1).mean(dim=1)

            if contrib_tensor.numel() == channels:
                return contrib_tensor.view(channels)

            # Fallback: if tensor already shaped (C,H,W), reduce as requested
            if contrib_tensor.dim() == 3 and contrib_tensor.shape[0] == channels:
                if self.flatten_strategy == 'element':
                    return contrib_tensor
                return contrib_tensor.view(channels, -1).mean(dim=1)

            return contrib_tensor

        if contrib_tensor.dim() == 1 and contrib_tensor.shape[0] == C:
            return contrib_tensor

        c = contrib_tensor.unsqueeze(0)
        if self.classifier_prefix_modules:
            for module in reversed(self.classifier_prefix_modules):
                if isinstance(module, nn.Linear):
                    c = torch.matmul(c, module.weight)
                elif isinstance(module, nn.ReLU):
                    c = torch.clamp(c, min=0.0)
                else:
                    # Dropout / other layers leave contributions unchanged in eval
                    pass
        c = c.squeeze(0)

        H, W = features_spatial.shape[2], features_spatial.shape[3]
        pooled_H, pooled_W = H, W
        if self.avgpool_layer is not None:
            with torch.no_grad():
                pooled = self.avgpool_layer(features_spatial)
            pooled_H, pooled_W = pooled.shape[2], pooled.shape[3]

        expected_spatial = C * H * W
        expected_pooled = C * pooled_H * pooled_W

        if c.numel() == expected_spatial:
            c = c.view(C, H, W)
        elif c.numel() == expected_pooled:
            c = c.view(C, pooled_H, pooled_W)
        else:
            # Fallback: average to channels if dimensions mismatch
            if c.numel() > C:
                reduced = c.view(C, -1).mean(dim=1)
                return reduced
            if c.shape[0] < C:
                return torch.nn.functional.pad(c, (0, C - c.shape[0]))
            return c

        if self.flatten_strategy == 'element':
            return c
        return c.view(C, -1).mean(dim=1)
    
    def __call__(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        return_features: bool = False
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Generate NObSP-CAM for an input image.
        
        Parameters
        ----------
        image : torch.Tensor
            Input image [B, 3, H, W]
        target_class : int, optional
            Target class for explanation (if None, uses predicted class)
        return_features : bool
            Whether to return intermediate features
            
        Returns
        -------
        dict
            CAM results with heatmaps and contributions
        """
        # Extract features
        features_spatial, classifier_input, feature_vector, logits = self.extract_features(image)
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(1).item()
        
        # Compute NObSP contributions
        contributions, coefficients = self.compute_nobsp_contributions(
            feature_vector,
            target_class
        )
        
        # Generate CAMs
        cam_results = self.generate_cam(features_spatial, contributions)
        
        # Upsample CAMs to original image size
        H_orig, W_orig = image.shape[2], image.shape[3]
        
        def upsample_cam(cam):
            cam_tensor = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            cam_upsampled = F.interpolate(
                cam_tensor,
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )
            return cam_upsampled.squeeze().cpu().numpy()
        
        # Prepare results
        results = {
            'heatmap': upsample_cam(cam_results['cam']),
            'heatmap_positive': upsample_cam(cam_results['cam_positive']),
            'heatmap_negative': upsample_cam(cam_results['cam_negative']),
            'heatmap_signed': upsample_cam(cam_results['cam_signed']),
            'heatmap_combined': upsample_cam(cam_results['cam_combined']),
            'contributions': cam_results['contributions'],
            'contributions_raw': cam_results['contributions_raw'],
            'predicted_class': logits.argmax(1).item(),
            'target_class': target_class,
            'features_shape': features_spatial.shape
        }
        
        if return_features:
            results['features_spatial'] = features_spatial
            results['classifier_input'] = classifier_input
            results['feature_vector'] = feature_vector
            results['logits'] = logits
        
        return results


def visualize_nobsp_cam(
    image: np.ndarray,
    results: dict,
    class_names: Optional[List[str]] = None,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize NObSP-CAM results.
    
    Parameters
    ----------
    image : np.ndarray
        Original image in [0, 1] range, shape [H, W, 3]
    results : dict
        Results from NObSPCAM
    class_names : list, optional
        List of class names for display
    alpha : float
        Overlay transparency
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Main heatmap overlay
    heatmap_colored = cm.jet(results['heatmap'])[:, :, :3]
    overlay = alpha * heatmap_colored + (1 - alpha) * image
    axes[0, 1].imshow(overlay)
    pred_class = results['predicted_class']
    target_class = results['target_class']
    title = f'NObSP-CAM\nPred: {pred_class}, Target: {target_class}'
    if class_names:
        title = f'NObSP-CAM\nPred: {class_names[pred_class]}\nTarget: {class_names[target_class]}'
    axes[0, 1].set_title(title)
    axes[0, 1].axis('off')
    
    # Heatmap only
    axes[0, 2].imshow(results['heatmap'], cmap='jet')
    axes[0, 2].set_title('Heatmap Only')
    axes[0, 2].axis('off')
    
    # Positive contributions
    heatmap_pos_colored = cm.jet(results['heatmap_positive'])[:, :, :3]
    overlay_pos = alpha * heatmap_pos_colored + (1 - alpha) * image
    axes[1, 0].imshow(overlay_pos)
    axes[1, 0].set_title('Positive Contributions')
    axes[1, 0].axis('off')
    
    # Negative contributions
    heatmap_neg_colored = cm.jet(results['heatmap_negative'])[:, :, :3]
    overlay_neg = alpha * heatmap_neg_colored + (1 - alpha) * image
    axes[1, 1].imshow(overlay_neg)
    axes[1, 1].set_title('Negative Contributions')
    axes[1, 1].axis('off')
    
    # Contribution distribution
    contributions = results['contributions']
    positive_mask = contributions > 0
    negative_mask = contributions < 0
    
    x = np.arange(len(contributions))
    axes[1, 2].bar(x[positive_mask], contributions[positive_mask], color='steelblue', alpha=0.7)
    axes[1, 2].bar(x[negative_mask], contributions[negative_mask], color='indianred', alpha=0.7)
    axes[1, 2].set_title(f'Channel Contributions\n({len(contributions)} channels)')
    axes[1, 2].set_xlabel('Channel Index')
    axes[1, 2].set_ylabel('Contribution')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add statistics
    pos_count = np.sum(positive_mask)
    neg_count = np.sum(negative_mask)
    max_contrib = np.max(np.abs(contributions))
    axes[1, 2].text(0.02, 0.98, f'Pos: {pos_count}, Neg: {neg_count}\nMax: {max_contrib:.3f}',
                    transform=axes[1, 2].transAxes, va='top', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
