"""
CNN-specific decomposition algorithms for NObSP.

These functions implement channel-based decomposition methods for CNNs,
allowing analysis from any intermediate layer to the output.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union

from .oblique import oblique_projection, oblique_projection_beta
from ..utils.tensor_ops import to_tensor, to_numpy


def build_forward_model(
    model: nn.Module,
    cut_layer_name: str,
    device: torch.device,
    return_pooled_features: bool = False
) -> nn.Module:
    """
    Build a forward model from a specified layer to the output.
    
    This creates a sub-network that starts from the specified layer
    and includes all subsequent layers until the output. For class-aware
    decomposition, it ensures proper transformation to 512-D pooled features
    before the final FC layer.
    
    IMPORTANT: The forward model can return both logits AND pooled features
    to avoid direct pooling artifacts in intermediate layers.
    
    Parameters
    ----------
    model : nn.Module
        Full CNN model
    cut_layer_name : str
        Name of the layer to start from (e.g., 'layer3.1.relu')
    device : torch.device
        Device for the model
    return_pooled_features : bool
        If True, returns 512-D pooled features; if False, returns class predictions
        
    Returns
    -------
    forward_model : nn.Module
        Sub-network from cut layer to output (or pooled features)
    """
    
    class ForwardModel(nn.Module):
        def __init__(self, full_model, start_layer_name):
            super().__init__()
            self.full_model = full_model
            self.start_layer_name = start_layer_name
            self.start_layer_found = False
            self.layers_after_cut = []
            
            # Identify architecture type
            self.is_resnet = hasattr(full_model, 'layer4')
            self.is_vgg = hasattr(full_model, 'features')
            self.is_efficientnet = hasattr(full_model, 'conv_head')
            
            # Build the forward path
            self._build_forward_path()
            
        def _build_forward_path(self):
            """Identify and store layers after the cut point."""
            if self.is_resnet:
                self._build_resnet_path()
            elif self.is_vgg:
                self._build_vgg_path()
            elif self.is_efficientnet:
                self._build_efficientnet_path()
            else:
                self._build_generic_path()
                
        def _build_resnet_path(self):
            """Build forward path for ResNet architectures."""
            # Parse layer name (e.g., 'layer3.1.relu' -> layer3, block 1, relu)
            parts = self.start_layer_name.split('.')
            
            # Determine which layers come after the cut
            layer_num = int(parts[0][-1]) if parts[0].startswith('layer') else -1
            
            if layer_num >= 0:
                # Build convolutional layers after the cut
                self.conv_layers = nn.Sequential()
                
                # If cutting within a layer, need to handle partial blocks
                if len(parts) > 1:
                    # Cutting within a layer - more complex handling needed
                    # For now, we'll start from the next full layer
                    layer_num += 1
                
                # Add subsequent layer blocks
                for i in range(layer_num, 5):  # ResNet has layer1-4
                    if i <= 4 and hasattr(self.full_model, f'layer{i}'):
                        self.conv_layers.add_module(
                            f'layer{i}', 
                            getattr(self.full_model, f'layer{i}')
                        )
                
                # Store pooling and FC separately for feature extraction
                self.avgpool = self.full_model.avgpool if hasattr(self.full_model, 'avgpool') else None
                self.fc = self.full_model.fc if hasattr(self.full_model, 'fc') else None
            else:
                # No valid layer number found
                self.conv_layers = nn.Sequential()
                self.avgpool = self.full_model.avgpool if hasattr(self.full_model, 'avgpool') else None
                self.fc = self.full_model.fc if hasattr(self.full_model, 'fc') else None
                    
        def _build_vgg_path(self):
            """Build forward path for VGG architectures."""
            # For VGG, we need to find the layer index in features
            found = False
            remaining_features = []
            
            for name, module in self.full_model.features.named_children():
                if found:
                    remaining_features.append((name, module))
                elif name == self.start_layer_name or f"features.{name}" == self.start_layer_name:
                    found = True
                    
            self.remaining_layers = nn.Sequential()
            self.conv_layers = nn.Sequential()  # Initialize for compatibility
            
            # Add remaining feature layers
            if remaining_features:
                features_seq = nn.Sequential()
                for name, module in remaining_features:
                    features_seq.add_module(name, module)
                self.remaining_layers.add_module('features', features_seq)
                
            # Add pooling and classifier
            if hasattr(self.full_model, 'avgpool'):
                self.remaining_layers.add_module('avgpool', self.full_model.avgpool)
            self.remaining_layers.add_module('flatten', nn.Flatten())
            if hasattr(self.full_model, 'classifier'):
                self.remaining_layers.add_module('classifier', self.full_model.classifier)
                
        def _build_efficientnet_path(self):
            """Build forward path for EfficientNet architectures."""
            # Similar to ResNet but with different layer names
            self.remaining_layers = nn.Sequential()
            self.conv_layers = nn.Sequential()  # Initialize for compatibility
            
            # Add remaining blocks after the cut
            # This is simplified - would need more sophisticated parsing for production
            if hasattr(self.full_model, 'conv_head'):
                self.remaining_layers.add_module('conv_head', self.full_model.conv_head)
            if hasattr(self.full_model, 'bn2'):
                self.remaining_layers.add_module('bn2', self.full_model.bn2)
            if hasattr(self.full_model, 'act2'):
                self.remaining_layers.add_module('act2', self.full_model.act2)
            if hasattr(self.full_model, 'global_pool'):
                self.remaining_layers.add_module('global_pool', self.full_model.global_pool)
            if hasattr(self.full_model, 'classifier'):
                self.remaining_layers.add_module('flatten', nn.Flatten())
                self.remaining_layers.add_module('classifier', self.full_model.classifier)
                
        def _build_generic_path(self):
            """Build forward path for generic architectures."""
            # For unknown architectures, try to build a sequential path
            self.remaining_layers = nn.Sequential()
            self.conv_layers = nn.Sequential()  # Initialize for compatibility
            
            # This would need custom implementation based on the specific model
            # For now, we'll just add common final layers
            if hasattr(self.full_model, 'avgpool'):
                self.remaining_layers.add_module('avgpool', self.full_model.avgpool)
            if hasattr(self.full_model, 'fc'):
                self.remaining_layers.add_module('flatten', nn.Flatten())
                self.remaining_layers.add_module('fc', self.full_model.fc)
            elif hasattr(self.full_model, 'classifier'):
                self.remaining_layers.add_module('flatten', nn.Flatten())
                self.remaining_layers.add_module('classifier', self.full_model.classifier)
                
        def forward(self, x, return_features=False):
            """
            Forward pass through the sub-network.
            
            Parameters
            ----------
            x : torch.Tensor
                Input tensor from the cut layer [B, C, H, W]
            return_features : bool
                If True, returns (logits, pooled_features)
                
            Returns
            -------
            output : torch.Tensor or tuple
                Model output (logits) [B, num_classes]
                If return_features=True, returns (logits, pooled_features)
            """
            if self.is_resnet:
                # Pass through remaining conv layers
                if hasattr(self, 'conv_layers') and len(self.conv_layers) > 0:
                    x = self.conv_layers(x)
                
                # Apply pooling
                if self.avgpool is not None:
                    x = self.avgpool(x)
                
                # Flatten to get pooled features
                pooled_features = torch.flatten(x, 1)  # [B, 512] for ResNet18
                
                # Apply FC for logits
                logits = self.fc(pooled_features) if self.fc is not None else pooled_features
                
                if return_features:
                    return logits, pooled_features
                return logits
            else:
                # Original behavior for other architectures
                return self.remaining_layers(x)
    
    # Create and return the forward model
    forward_model = ForwardModel(model, cut_layer_name).to(device)
    forward_model.eval()
    return forward_model


def decompose_alpha_cnn(
    X: torch.Tensor,
    y_pred: torch.Tensor,
    forward_model: nn.Module,
    device: torch.device,
    regularization: float = 1e-6,
    batch_size: int = 32,
    pooled_feature_dim: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute alpha coefficient decomposition for CNN channels.
    
    This method treats each channel as a feature and computes alpha
    coefficients that capture channel contributions through the network.
    
    Parameters
    ----------
    X : torch.Tensor
        Input feature maps from cut layer [n_samples, n_channels, H, W]
    y_pred : torch.Tensor
        Model predictions (logits) [n_samples, n_classes]
    forward_model : nn.Module
        Sub-network from cut layer to output
    device : torch.device
        Device for computation
    regularization : float
        Regularization parameter
    batch_size : int
        Batch size for processing (to manage memory)
        
    Returns
    -------
    alpha : np.ndarray
        Alpha coefficients [n_channels, hidden_size * n_classes]
    contributions : np.ndarray
        Channel contributions [n_samples, n_channels, n_classes]
    """
    n_samples, n_channels, H, W = X.shape
    n_classes = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
    
    if n_classes == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Determine hidden_size - should always be the final pooled dimension
    # With proper forward transformation, all layers use 512-D space
    if pooled_feature_dim is not None:
        hidden_size = pooled_feature_dim  # Use provided dimension (512 for ResNet18)
    else:
        # Fallback - assume 512 for ResNet18
        hidden_size = 512 if n_channels <= 512 else n_channels
        import warnings
        warnings.warn(
            f"pooled_feature_dim not provided. Using {hidden_size} as hidden size."
        )
    
    # Initialize storage for FULL coefficients (channel-centric structure)
    # Shape: (n_channels, hidden_size * n_classes) to match nobsp_cam.py
    Alpha = torch.zeros(n_channels, hidden_size * n_classes, device=device, dtype=torch.float32)
    contributions = np.zeros((n_samples, n_channels, n_classes))
    
    # Process each channel
    print(f"Processing {n_channels} channels × {n_classes} classes = {n_channels * n_classes} coefficients...")
    for i in range(n_channels):
        if i % 50 == 0:
            print(f"  Channel {i}/{n_channels}...")
        # Create target input (only channel i active)
        X_target = torch.zeros_like(X)
        X_target[:, i, :, :] = X[:, i, :, :]
        
        # Create reference input (all channels except i)
        X_reference = X.clone()
        X_reference[:, i, :, :] = 0
        
        # Get properly transformed features through the forward model
        # This avoids pooling artifacts by using the full forward transformation
        with torch.no_grad():
            # For proper decomposition, we need features in the final pooled space
            # This means passing through all subsequent layers
            if hasattr(forward_model, 'forward'):
                # Get both logits and pooled features
                _, X_target_sub = forward_model(X_target, return_features=True)
                _, X_reference_sub = forward_model(X_reference, return_features=True)
                # Now X_target_sub and X_reference_sub are properly in 512-D space
            else:
                # Fallback for compatibility
                target_pooled = torch.nn.functional.adaptive_avg_pool2d(X_target, (1, 1))
                reference_pooled = torch.nn.functional.adaptive_avg_pool2d(X_reference, (1, 1))
                X_target_sub = torch.flatten(target_pooled, 1)
                X_reference_sub = torch.flatten(reference_pooled, 1)
        
        # Center the outputs
        X_target_sub_centered = X_target_sub - X_target_sub.mean(dim=0)
        X_reference_sub_centered = X_reference_sub - X_reference_sub.mean(dim=0)
        
        # Compute alpha coefficients for each class
        for j in range(n_classes):
            y_target = y_pred[:, j]
            
            # Use oblique projection to get coefficients
            try:
                # Compute using standard least squares with regularization
                # We're solving: X_target_sub_centered @ alpha = y_target
                XtX = X_target_sub_centered.T @ X_target_sub_centered
                XtX_reg = XtX + regularization * torch.eye(XtX.shape[0], device=device)
                Xty = X_target_sub_centered.T @ y_target
                alpha_j = torch.linalg.solve(XtX_reg, Xty)
                
                # Store coefficients in channel-centric structure
                # Ensure alpha_j is properly shaped
                if alpha_j.ndim > 1:
                    alpha_j = alpha_j.squeeze()
                if alpha_j.ndim == 0:
                    alpha_j = alpha_j.unsqueeze(0)
                
                # Ensure correct size
                if alpha_j.shape[0] != hidden_size:
                    if alpha_j.shape[0] == 1:
                        alpha_j = alpha_j.repeat(hidden_size)
                    else:
                        # Handle size mismatch gracefully
                        alpha_j = alpha_j[:hidden_size] if alpha_j.shape[0] > hidden_size else torch.nn.functional.pad(alpha_j, (0, hidden_size - alpha_j.shape[0]))
                
                # Store in channel-centric format: Alpha[channel_idx, class_coefficients]
                Alpha[i, j*hidden_size:(j+1)*hidden_size] = alpha_j
                
                # Compute contribution
                contributions[:, i, j] = to_numpy(X_target_sub @ alpha_j)
                
            except Exception as e:
                print(f"Warning: Failed to compute alpha for channel {i}, class {j}: {e}")
                # Set to zero in the correct position
                Alpha[:, coef_idx] = 0
                contributions[:, i, j] = 0
    
    return to_numpy(Alpha), contributions


def decompose_beta_cnn(
    X: torch.Tensor,
    y_pred: torch.Tensor,
    forward_model: nn.Module,
    device: torch.device,
    regularization: float = 1e-6,
    batch_size: int = 32,
    pooled_feature_dim: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute beta coefficient decomposition for CNN channels using partial regression.
    
    The beta method isolates each channel's unique contribution after
    accounting for all other channels' effects.
    
    Parameters
    ----------
    X : torch.Tensor
        Input feature maps from cut layer [n_samples, n_channels, H, W]
    y_pred : torch.Tensor
        Model predictions (logits) [n_samples, n_classes]
    forward_model : nn.Module
        Sub-network from cut layer to output
    device : torch.device
        Device for computation
    regularization : float
        Regularization parameter
    batch_size : int
        Batch size for processing (to manage memory)
        
    Returns
    -------
    beta : np.ndarray
        Beta coefficients [n_channels, hidden_size * n_classes]
    contributions : np.ndarray
        Channel contributions [n_samples, n_channels, n_classes]
    """
    n_samples, n_channels, H, W = X.shape
    n_classes = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
    
    if n_classes == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Determine hidden_size - should always be the final pooled dimension
    # With proper forward transformation, all layers use 512-D space
    if pooled_feature_dim is not None:
        hidden_size = pooled_feature_dim  # Use provided dimension (512 for ResNet18)
    else:
        # Fallback - assume 512 for ResNet18
        hidden_size = 512 if n_channels <= 512 else n_channels
        import warnings
        warnings.warn(
            f"pooled_feature_dim not provided. Using {hidden_size} as hidden size."
        )
    
    # Initialize storage for FULL coefficients (channel-centric structure)
    # Shape: (n_channels, hidden_size * n_classes) to match nobsp_cam.py
    Beta = torch.zeros(n_channels, hidden_size * n_classes, device=device, dtype=torch.float32)
    contributions = np.zeros((n_samples, n_channels, n_classes))
    
    # Process each channel
    print(f"Processing {n_channels} channels × {n_classes} classes = {n_channels * n_classes} coefficients...")
    for i in range(n_channels):
        if i % 50 == 0:
            print(f"  Channel {i}/{n_channels}...")
        # Create target input (only channel i active)
        X_target = torch.zeros_like(X)
        X_target[:, i, :, :] = X[:, i, :, :]
        
        # Create reference input (all channels except i)
        X_reference = X.clone()
        X_reference[:, i, :, :] = 0
        
        # Get properly transformed features through the forward model
        # This avoids pooling artifacts by using the full forward transformation
        with torch.no_grad():
            # For proper decomposition, we need features in the final pooled space
            # This means passing through all subsequent layers
            if hasattr(forward_model, 'forward'):
                # Get both logits and pooled features
                _, X_target_sub = forward_model(X_target, return_features=True)
                _, X_reference_sub = forward_model(X_reference, return_features=True)
                # Now X_target_sub and X_reference_sub are properly in 512-D space
            else:
                # Fallback for compatibility
                target_pooled = torch.nn.functional.adaptive_avg_pool2d(X_target, (1, 1))
                reference_pooled = torch.nn.functional.adaptive_avg_pool2d(X_reference, (1, 1))
                X_target_sub = torch.flatten(target_pooled, 1)
                X_reference_sub = torch.flatten(reference_pooled, 1)
        
        # Center the outputs
        X_target_sub_centered = X_target_sub - X_target_sub.mean(dim=0)
        X_reference_sub_centered = X_reference_sub - X_reference_sub.mean(dim=0)
        
        # Compute beta coefficients for each class using partial regression
        for j in range(n_classes):
            y_target = y_pred[:, j]
            y_target_centered = y_target - y_target.mean()
            
            # Use oblique projection beta method
            beta_j = oblique_projection_beta(
                X_target_sub_centered, 
                X_reference_sub_centered,
                y_target_centered, 
                device, 
                lambda_reg=regularization
            )
            
            # Store coefficients in channel-centric structure
            # Each channel gets a row, coefficients for each class are stored consecutively
            # Ensure beta_j is properly shaped
            if beta_j.ndim > 1:
                beta_j = beta_j.squeeze()
            if beta_j.ndim == 0:
                beta_j = beta_j.unsqueeze(0)
            
            # Ensure correct size
            if beta_j.shape[0] != hidden_size:
                if beta_j.shape[0] == 1:
                    beta_j = beta_j.repeat(hidden_size)
                else:
                    # This shouldn't happen, but handle it gracefully
                    print(f"Warning: coefficient size mismatch for channel {i}, class {j}")
                    beta_j = beta_j[:hidden_size] if beta_j.shape[0] > hidden_size else torch.nn.functional.pad(beta_j, (0, hidden_size - beta_j.shape[0]))
            
            # Store in channel-centric format: Beta[channel_idx, class_coefficients]  
            Beta[i, j*hidden_size:(j+1)*hidden_size] = beta_j
            
            # Compute contribution (apply to uncentered for proper contribution)
            contributions[:, i, j] = to_numpy(X_target_sub @ beta_j)
    
    return to_numpy(Beta), contributions


def apply_channel_coefficients(
    X: torch.Tensor,
    forward_model: nn.Module,
    coefficients: np.ndarray,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Apply stored channel coefficients to new data.
    
    Parameters
    ----------
    X : torch.Tensor
        Input feature maps [n_samples, n_channels, H, W]
    forward_model : nn.Module
        Sub-network from cut layer to output
    coefficients : np.ndarray
        Stored coefficients [n_channels, n_classes]
    device : torch.device
        Device for computation
    batch_size : int
        Batch size for processing
        
    Returns
    -------
    contributions : np.ndarray
        Channel contributions [n_samples, n_channels, n_classes]
    """
    n_samples, n_channels, H, W = X.shape
    n_classes = coefficients.shape[1]
    
    # Initialize storage
    contributions = np.zeros((n_samples, n_channels, n_classes))
    coef_tensor = to_tensor(coefficients, device)
    
    # Process each channel
    for i in range(n_channels):
        # Create input with only channel i active
        X_single = torch.zeros_like(X)
        X_single[:, i, :, :] = X[:, i, :, :]
        
        # Process in batches
        channel_outputs = []
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            
            with torch.no_grad():
                output = forward_model(X_single[batch_start:batch_end])
                channel_outputs.append(output)
        
        # Concatenate outputs
        X_transformed = torch.cat(channel_outputs, dim=0)
        
        # Apply coefficients for each class
        for j in range(n_classes):
            # Simple multiplication with the coefficient
            contributions[:, i, j] = to_numpy(X_transformed[:, j] * coef_tensor[i, j])
    
    return contributions