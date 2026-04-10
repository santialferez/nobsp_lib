"""Adapters that make ordinary PyTorch models compatible with the NObSP core."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def resolve_module(root: nn.Module, module: str | nn.Module | None) -> nn.Module | None:
    """Resolve a module from a dotted path or return it unchanged."""
    if module is None:
        return None
    if isinstance(module, nn.Module):
        return module

    current = root
    for part in module.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def find_last_linear(model: nn.Module) -> nn.Linear:
    """Return the final linear layer found in a module tree."""
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError(
            "Could not find a Linear layer in the model. "
            "Pass feature_layer explicitly or provide a model with a linear head."
        )
    return last_linear


def select_tensor_output(output: Any, selector: int | str | Callable[[Any], Any] | None) -> torch.Tensor:
    """Extract a tensor output from common model return formats."""
    if callable(selector):
        output = selector(output)
    elif isinstance(selector, int):
        output = output[selector]
    elif isinstance(selector, str):
        output = output[selector]

    if torch.is_tensor(output):
        return output

    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return item
        raise ValueError("Could not find a tensor inside tuple/list model output.")

    if isinstance(output, dict):
        for key in ("logits", "predictions", "prediction", "output", "y"):
            item = output.get(key)
            if torch.is_tensor(item):
                return item
        for item in output.values():
            if torch.is_tensor(item):
                return item
        raise ValueError("Could not find a tensor inside dict model output.")

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def looks_like_probabilities(output: torch.Tensor) -> bool:
    """Heuristic to detect probability-like classification outputs."""
    if output.ndim != 2 or output.shape[1] < 2:
        return False
    if torch.any(output < -1e-5) or torch.any(output > 1.0 + 1e-5):
        return False
    row_sums = output.sum(dim=1)
    return torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3)


class HookedTabularAdapter(nn.Module):
    """Wrap a standard PyTorch model into the tuple format expected by NObSP."""

    def __init__(
        self,
        model: nn.Module,
        task: str,
        *,
        feature_layer: str | nn.Module | None = None,
        capture: str = "auto",
        output_kind: str = "auto",
        output_selector: int | str | Callable[[Any], Any] | None = None,
    ):
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'.")
        if capture not in {"auto", "input", "output"}:
            raise ValueError("capture must be 'auto', 'input', or 'output'.")
        if output_kind not in {"auto", "logits", "probabilities", "predictions"}:
            raise ValueError(
                "output_kind must be 'auto', 'logits', 'probabilities', or 'predictions'."
            )

        self.model = model
        self.task = task
        self.output_kind = output_kind
        self.output_selector = output_selector
        self._captured_features: torch.Tensor | None = None

        target_module = resolve_module(model, feature_layer) if feature_layer is not None else find_last_linear(model)
        if target_module is None:
            raise ValueError("feature_layer could not be resolved.")

        hook_mode = capture
        if hook_mode == "auto":
            hook_mode = "input" if isinstance(target_module, nn.Linear) else "output"
        self.capture_mode = hook_mode

        if hook_mode == "input":
            self._hook_handle = target_module.register_forward_pre_hook(self._capture_input)
        else:
            self._hook_handle = target_module.register_forward_hook(self._capture_output)

    def _capture_input(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        features = inputs[0]
        if not torch.is_tensor(features):
            raise TypeError("Captured feature input is not a tensor.")
        self._captured_features = self._flatten_features(features)

    def _capture_output(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Captured feature output is not a tensor.")
        self._captured_features = self._flatten_features(output)

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(1)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def _normalize_classification_output(
        self,
        output_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_tensor.ndim == 1:
            output_tensor = output_tensor.unsqueeze(1)

        if output_tensor.shape[1] == 1:
            score = output_tensor.squeeze(1)
            if self.output_kind == "probabilities":
                prob_pos = torch.clamp(score, 1e-6, 1 - 1e-6)
                probabilities = torch.stack([1.0 - prob_pos, prob_pos], dim=1)
                logits = torch.log(probabilities)
            else:
                logits = torch.stack([-score, score], dim=1)
                probabilities = F.softmax(logits, dim=1)
            return probabilities, logits

        if self.output_kind == "probabilities":
            probabilities = torch.clamp(output_tensor, 1e-6, 1.0)
            logits = torch.log(probabilities)
            return probabilities, logits

        if self.output_kind == "auto" and looks_like_probabilities(output_tensor):
            probabilities = torch.clamp(output_tensor, 1e-6, 1.0)
            logits = torch.log(probabilities)
            return probabilities, logits

        logits = output_tensor
        probabilities = F.softmax(logits, dim=1)
        return probabilities, logits

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._captured_features = None
        raw_output = self.model(x)
        output_tensor = select_tensor_output(raw_output, self.output_selector)
        features = self._captured_features
        if features is None:
            raise RuntimeError(
                "Could not capture penultimate features from the wrapped model. "
                "Check feature_layer/capture settings."
            )

        if self.task == "classification":
            probabilities, logits = self._normalize_classification_output(output_tensor)
            return probabilities, features, logits

        predictions = output_tensor.float()
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(1)
        return predictions, features
