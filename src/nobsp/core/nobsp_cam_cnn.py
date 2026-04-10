"""
NObSP-CAM for CNNs: Generalized to work from any layer.

This module extends NObSP-CAM to generate class activation maps 
from any intermediate CNN layer, not just the final FC layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple, Union, Dict, List
import warnings

# Import decomposition functions
from nobsp.core.decompose_cnn import (
    decompose_alpha_cnn, 
    decompose_beta_cnn,
    build_forward_model,
    apply_channel_coefficients
)
from nobsp.utils.tensor_ops import to_tensor, to_numpy


class NObSPCAM_CNN:
    """
    Generalized NObSP-CAM for CNN interpretability from any layer.
    
    This implementation can analyze CNNs starting from any intermediate
    layer, computing channel contributions through the rest of the network.
    
    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained CNN model (e.g., ResNet, VGG)
    cut_layer : str
        Name/path of layer to start analysis from (e.g., 'layer3.1.relu')
    target_layer : str, optional
        Name/path of layer to extract spatial features from for CAM visualization
        If None, uses the same as cut_layer
    method : str
        Decomposition method: 'alpha' or 'beta' (default: 'beta')
    device : str or torch.device, optional
        Device for computation (auto-detected if None)
    regularization : float
        Regularization parameter for decomposition
    batch_size : int
        Batch size for processing (to manage memory)
    """
    
    def __init__(
        self,
        model: nn.Module,
        cut_layer: str,
        target_layer: Optional[str] = None,
        method: str = 'beta',
        device: Optional[Union[str, torch.device]] = None,
        regularization: float = 1e-6,
        batch_size: int = 16
    ):
        # Store original model for reference
        self.original_model = model.eval()
        self.cut_layer_name = cut_layer
        self.target_layer_name = target_layer or cut_layer
        self.method = method
        self.regularization = regularization
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = (torch.device('cuda') if torch.cuda.is_available()
                          else torch.device('mps') if torch.backends.mps.is_available()
                          else torch.device('cpu'))
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        self.model = self.original_model.to(self.device)
        
        # Build forward model from the (potentially reduced) model
        self.forward_model = build_forward_model(self.model, self.cut_layer_name, self.device)
        
        # Store layer information for proper multi-layer handling
        self._detect_layer_info()
        
        # Identify layers for feature extraction
        self.cut_layer = self._get_layer_by_name(self.cut_layer_name)
        self.target_layer = self._get_layer_by_name(self.target_layer_name)
        
        # Storage for features from hooks
        self.cut_features = None
        self.target_features = None
        self.hook_handles = []
        
        # Cache for coefficients
        self.cached_coefficients = None
        self.calibration_metadata = {}
        
    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Get a layer from the model by its name."""
        parts = layer_name.split('.')
        layer = self.model
        
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
                
        return layer
    
    def _detect_layer_info(self):
        """Detect layer information for proper multi-layer handling."""
        # Dynamically detect pooled feature dimension from FC layer
        if hasattr(self.model, 'fc'):
            # ResNet style
            self.pooled_feature_dim = self.model.fc.in_features
        elif hasattr(self.model, 'classifier'):
            # VGG/AlexNet style
            if isinstance(self.model.classifier, nn.Sequential):
                # Find first Linear layer
                for module in self.model.classifier:
                    if isinstance(module, nn.Linear):
                        self.pooled_feature_dim = module.in_features
                        break
            else:
                self.pooled_feature_dim = self.model.classifier.in_features
        elif hasattr(self.model, 'head'):
            # EfficientNet style
            self.pooled_feature_dim = self.model.head.in_features
        else:
            # Fallback - try to detect from forward pass
            warnings.warn("Could not detect pooled feature dimension, defaulting to 512")
            self.pooled_feature_dim = 512
            
        # Detect architecture type for proper handling
        self.is_resnet = hasattr(self.model, 'layer4')
        self.is_vgg = hasattr(self.model, 'features') and hasattr(self.model, 'classifier')
        self.is_densenet = hasattr(self.model, 'features') and hasattr(self.model, 'norm5')
        
        # For ResNet-specific layer detection (backward compatibility)
        if self.is_resnet:
            if 'layer4' in self.cut_layer_name:
                self.cut_layer_num = 4
            elif 'layer3' in self.cut_layer_name:
                self.cut_layer_num = 3
            elif 'layer2' in self.cut_layer_name:
                self.cut_layer_num = 2
            elif 'layer1' in self.cut_layer_name:
                self.cut_layer_num = 1
            else:
                self.cut_layer_num = -1
        else:
            self.cut_layer_num = -1
    
    def _hook_cut_features(self, module, input, output):
        """Hook to capture features from cut layer."""
        self.cut_features = output.detach().clone()
        
    def _hook_target_features(self, module, input, output):
        """Hook to capture features from target layer (for visualization)."""
        self.target_features = output.detach().clone()
    
    def extract_features(
        self, 
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from cut and target layers.
        
        Parameters
        ----------
        images : torch.Tensor
            Input images [B, 3, H, W]
            
        Returns
        -------
        cut_features : torch.Tensor
            Features from cut layer [B, C_cut, H_cut, W_cut]
        target_features : torch.Tensor
            Features from target layer [B, C_target, H_target, W_target]
        predictions : torch.Tensor
            Model predictions [B, num_classes]
        """
        # Clear previous hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        # Register hooks
        handle_cut = self.cut_layer.register_forward_hook(self._hook_cut_features)
        self.hook_handles.append(handle_cut)
        
        if self.target_layer_name != self.cut_layer_name:
            handle_target = self.target_layer.register_forward_hook(self._hook_target_features)
            self.hook_handles.append(handle_target)
        
        # Forward pass through full model to trigger hooks
        with torch.no_grad():
            predictions = self.model(images.to(self.device))
        
        # Get features
        cut_features = self.cut_features
        target_features = self.target_features if self.target_layer_name != self.cut_layer_name else cut_features
        
        # Clear storage
        self.cut_features = None
        self.target_features = None
        
        # Remove hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        return cut_features, target_features, predictions
    
    def calibrate(
        self, 
        calibration_loader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> None:
        """
        Calibrate NObSP-CAM by computing coefficients on a dataset.
        
        This computes channel coefficients using multiple samples for
        meaningful oblique projections.
        
        Parameters
        ----------
        calibration_loader : DataLoader
            DataLoader with calibration images
        max_samples : int, optional
            Maximum number of samples to use (None = use all)
        verbose : bool
            Print progress information
        """
        if verbose:
            print(f"Calibrating NObSP-CAM from layer '{self.cut_layer_name}'...")
            
        all_cut_features = []
        all_predictions = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calibration_loader):
                if max_samples and samples_processed >= max_samples:
                    break
                    
                # Extract features
                cut_features, _, predictions = self.extract_features(images)
                
                # Limit samples if needed
                remaining = max_samples - samples_processed if max_samples else len(images)
                if remaining < len(images):
                    cut_features = cut_features[:remaining]
                    predictions = predictions[:remaining]
                
                all_cut_features.append(cut_features.cpu())
                all_predictions.append(predictions.cpu())
                
                samples_processed += len(predictions)
                
                if verbose and batch_idx % 10 == 0:
                    print(f"  Processed {samples_processed} samples...")
        
        # Concatenate all features
        all_cut_features = torch.cat(all_cut_features, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        
        if verbose:
            print(f"  Computing NObSP coefficients on {all_cut_features.shape[0]} samples...")
            print(f"  Feature shape: {all_cut_features.shape}")
        
        model_for_decompose = self.forward_model

        if verbose:
            print(f"  Using classifier with {all_predictions.shape[1]} classes")
        
        # Compute coefficients using CNN decomposition
        if self.method == 'beta':
            coefficients, _ = decompose_beta_cnn(
                X=all_cut_features.to(self.device),
                y_pred=all_predictions.to(self.device),
                forward_model=model_for_decompose,
                device=self.device,
                regularization=self.regularization,
                batch_size=self.batch_size,
                pooled_feature_dim=self.pooled_feature_dim
            )
        else:  # alpha
            coefficients, _ = decompose_alpha_cnn(
                X=all_cut_features.to(self.device),
                y_pred=all_predictions.to(self.device),
                forward_model=model_for_decompose,
                device=self.device,
                regularization=self.regularization,
                batch_size=self.batch_size,
                pooled_feature_dim=self.pooled_feature_dim
            )
        
        # Cache the coefficients
        self.cached_coefficients = coefficients
        
        # Store calibration metadata
        self.calibration_metadata = {
            'cut_layer': self.cut_layer_name,
            'target_layer': self.target_layer_name,
            'method': self.method,
            'samples': all_cut_features.shape[0],
            'channels': all_cut_features.shape[1],
            'spatial_dims': all_cut_features.shape[2:],
            'classes': all_predictions.shape[1],
        }
        
        if verbose:
            print(f"✓ Calibration complete!")
            print(f"  Coefficients shape: {coefficients.shape}")
            print(f"  Channels: {all_cut_features.shape[1]}")
            print(f"  Classes: {all_predictions.shape[1]}")
    
    def compute_channel_contributions(
        self,
        cut_features: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute channel contributions for given features.
        
        Parameters
        ----------
        cut_features : torch.Tensor
            Features from cut layer [B, C, H, W]
        target_class : int, optional
            Target class for explanation (if None, uses predicted)
            For reduced models, this should be the full ImageNet index.
            
        Returns
        -------
        contributions : np.ndarray
            Channel contributions [C,]
        """
        model_for_contrib = self.forward_model
        
        if self.cached_coefficients is None:
            warnings.warn(
                "NObSP-CAM not calibrated! Call .calibrate() first. "
                "Computing on single sample will give poor results.",
                UserWarning
            )
            
            # Get predictions
            with torch.no_grad():
                predictions = model_for_contrib(cut_features)
            
            # Compute coefficients on single sample
            if self.method == 'beta':
                coefficients, contributions_full = decompose_beta_cnn(
                    X=cut_features,
                    y_pred=predictions,
                    forward_model=model_for_contrib,
                    device=self.device,
                    regularization=self.regularization,
                    batch_size=self.batch_size,
                    pooled_feature_dim=self.pooled_feature_dim
                )
            else:
                coefficients, contributions_full = decompose_alpha_cnn(
                    X=cut_features,
                    y_pred=predictions,
                    forward_model=model_for_contrib,
                    device=self.device,
                    regularization=self.regularization,
                    batch_size=self.batch_size,
                    pooled_feature_dim=self.pooled_feature_dim
                )
            
            # Get target class
            if target_class is None:
                target_class = predictions.argmax(1).item()
            
            contributions = contributions_full[0, :, target_class]
            
        else:
            # Use cached coefficients
            coefficients = self.cached_coefficients
            
            # Get predictions for target class determination
            with torch.no_grad():
                predictions = model_for_contrib(cut_features)
            
            if target_class is None:
                target_class = predictions.argmax(1).item()
            
            coef_idx = target_class
            
            # Get contributions for target class
            n_channels = cut_features.shape[1]
            contributions = np.zeros(n_channels)
            
            # Apply coefficients to get contributions (channel-centric structure)
            # Determine hidden_size from coefficients shape
            # coefficients shape: (n_channels, hidden_size * n_classes)
            n_classes = predictions.shape[1]
            hidden_size = coefficients.shape[1] // n_classes
            
            for i in range(n_channels):
                # Isolate channel i
                X_single = torch.zeros_like(cut_features)
                X_single[:, i, :, :] = cut_features[:, i, :, :]
                
                # Get properly transformed features through forward model
                with torch.no_grad():
                    # Use the forward model to properly transform features
                    if hasattr(self.forward_model, 'forward'):
                        _, X_trans = self.forward_model(X_single, return_features=True)
                        # X_trans is now properly in 512-D space
                    else:
                        # Fallback for compatibility
                        pooled = torch.nn.functional.adaptive_avg_pool2d(X_single, (1, 1))
                        X_trans = torch.flatten(pooled, 1)
                
                # Get the appropriate coefficient vector for channel i and class coef_idx
                # New structure: coefficients[channel_idx, class_coefficients]
                coef_vector = coefficients[i, coef_idx*hidden_size:(coef_idx+1)*hidden_size]
                
                # Contribution is X_trans @ coef_vector
                contributions[i] = np.dot(X_trans[0].cpu().numpy(), coef_vector)
        
        return contributions
    
    def generate_cam(
        self,
        target_features: torch.Tensor,
        contributions: np.ndarray
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Generate CAM by combining spatial features with channel contributions.
        
        Parameters
        ----------
        target_features : torch.Tensor
            Spatial features for visualization [B, C, H, W]
        contributions : np.ndarray
            Channel contributions [C,]
            
        Returns
        -------
        dict
            Dictionary containing different CAM visualizations
        """
        B, C, H, W = target_features.shape
        
        # Handle channel mismatch (cut and target layers may have different channels)
        if len(contributions) != C:
            # Need to interpolate or map contributions
            # For now, we'll use the minimum channels
            min_channels = min(len(contributions), C)
            contributions = contributions[:min_channels]
            target_features = target_features[:, :min_channels, :, :]
            C = min_channels
        
        # Convert contributions to tensor
        contrib_tensor = torch.from_numpy(contributions).float().to(self.device)
        
        # Separate positive and negative contributions
        contrib_pos = contrib_tensor.clamp_min(0)
        contrib_neg = contrib_tensor.clamp_max(0).abs()
        
        # Generate weighted CAMs
        # Reshape for matrix multiplication
        features_flat = target_features[0].reshape(C, H*W).T  # [H*W, C]
        
        # Compute CAMs
        if contrib_tensor.abs().sum() > 0:
            cam_avg = torch.matmul(features_flat, contrib_tensor.abs()) / contrib_tensor.abs().sum()
        else:
            cam_avg = torch.zeros(H*W).to(self.device)
        
        if contrib_pos.sum() > 0:
            cam_pos = torch.matmul(features_flat, contrib_pos) / contrib_pos.sum()
        else:
            cam_pos = torch.zeros(H*W).to(self.device)
        
        if contrib_neg.sum() > 0:
            cam_neg = torch.matmul(features_flat, contrib_neg) / contrib_neg.sum()
        else:
            cam_neg = torch.zeros(H*W).to(self.device)
        
        # Reshape back to spatial dimensions
        cam_avg = cam_avg.view(H, W)
        cam_pos = cam_pos.view(H, W)
        cam_neg = cam_neg.view(H, W)
        
        # Normalize to [0, 1]
        def normalize_cam(cam):
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                return (cam - cam_min) / (cam_max - cam_min)
            return torch.zeros_like(cam)
        
        return {
            'cam': normalize_cam(cam_avg),
            'cam_positive': normalize_cam(cam_pos),
            'cam_negative': normalize_cam(cam_neg),
            'contributions': contributions,
            'contributions_raw': contrib_tensor.cpu().numpy()
        }
    
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
            Target class for explanation (if None, uses predicted)
        return_features : bool
            Whether to return intermediate features
            
        Returns
        -------
        dict
            CAM results with heatmaps and contributions
        """
        # Extract features
        cut_features, target_features, predictions = self.extract_features(image)
        
        # Determine target class
        if target_class is None:
            target_class = predictions.argmax(1).item()
            # If using reduced model, this is already in full ImageNet space
        
        # Compute channel contributions
        contributions = self.compute_channel_contributions(cut_features, target_class)
        
        # Generate CAMs using target layer features
        cam_results = self.generate_cam(target_features, contributions)
        
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
            'contributions': cam_results['contributions'],
            'contributions_raw': cam_results['contributions_raw'],
            'predicted_class': predictions.argmax(1).item(),
            'target_class': target_class,
            'cut_layer': self.cut_layer_name,
            'target_layer': self.target_layer_name,
            'cut_features_shape': cut_features.shape,
            'target_features_shape': target_features.shape
        }
        
        if return_features:
            results['cut_features'] = cut_features
            results['target_features'] = target_features
            results['predictions'] = predictions
        
        return results
    
    def save_coefficients(self, filepath: str):
        """Save calibration coefficients and metadata to disk."""
        if self.cached_coefficients is None:
            raise ValueError("No coefficients to save. Run calibrate() first.")
        
        save_data = {
            'coefficients': self.cached_coefficients,
            'metadata': self.calibration_metadata
        }
        
        np.savez_compressed(filepath, **save_data)
        print(f"Saved coefficients to {filepath}")
    
    def load_coefficients(self, filepath: str):
        """Load calibration coefficients and metadata from disk."""
        data = np.load(filepath, allow_pickle=True)
        
        self.cached_coefficients = data['coefficients']
        if 'metadata' in data:
            self.calibration_metadata = data['metadata'].item()

        if self.calibration_metadata and any(
            key in self.calibration_metadata for key in ('selected_classes', 'is_reduced', 'model_type')
        ):
            raise ValueError(
                "Loaded coefficients were generated with the legacy class-reduction "
                "workflow. Please recalibrate using the current NObSP-CAM API."
            )

        print(f"Loaded coefficients from {filepath}")
        print(f"  Shape: {self.cached_coefficients.shape}")
        if self.calibration_metadata:
            print(f"  Metadata: {self.calibration_metadata}")


def visualize_layer_comparison(
    image: np.ndarray,
    results_dict: Dict[str, dict],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize NObSP-CAM results from multiple layers.
    
    Parameters
    ----------
    image : np.ndarray
        Original image in [0, 1] range
    results_dict : dict
        Dictionary mapping layer names to their CAM results
    class_names : list, optional
        List of class names
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    n_layers = len(results_dict)
    fig, axes = plt.subplots(3, n_layers + 1, figsize=figsize)
    
    # Original image in first column
    for i in range(3):
        axes[i, 0].imshow(image)
        if i == 0:
            axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
    
    # CAMs for each layer
    for col_idx, (layer_name, results) in enumerate(results_dict.items(), 1):
        # Main CAM
        heatmap = results['heatmap']
        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        overlay = 0.5 * heatmap_colored + 0.5 * image
        axes[0, col_idx].imshow(overlay)
        axes[0, col_idx].set_title(f'{layer_name}\nMain CAM')
        axes[0, col_idx].axis('off')
        
        # Positive contributions
        heatmap_pos = results['heatmap_positive']
        heatmap_pos_colored = cm.jet(heatmap_pos)[:, :, :3]
        overlay_pos = 0.5 * heatmap_pos_colored + 0.5 * image
        axes[1, col_idx].imshow(overlay_pos)
        axes[1, col_idx].set_title('Positive')
        axes[1, col_idx].axis('off')
        
        # Negative contributions
        heatmap_neg = results['heatmap_negative']
        heatmap_neg_colored = cm.jet(heatmap_neg)[:, :, :3]
        overlay_neg = 0.5 * heatmap_neg_colored + 0.5 * image
        axes[2, col_idx].imshow(overlay_neg)
        axes[2, col_idx].set_title('Negative')
        axes[2, col_idx].axis('off')
    
    plt.suptitle('NObSP-CAM Layer Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
