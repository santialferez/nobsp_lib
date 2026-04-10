"""Community-facing vision API built on top of NObSPVision."""

from __future__ import annotations

from typing import Any, Literal

import torch

from nobsp import NObSPVision
from nobsp.utils.device import auto_detect_device


class EasyVisionExplainer:
    """Calibrate NObSP-CAM and generate heatmaps with a small API."""

    def __init__(
        self,
        method: str = "beta",
        target_layer: str | None = None,
        device: str | torch.device | None = None,
        regularization: float = 1e-6,
        flatten_strategy: str = "channel",
        decomposition_space: Literal["hidden", "classifier_input"] = "classifier_input",
        default_heatmap_mode: Literal["positive", "negative", "mixed"] = "positive",
    ):
        if default_heatmap_mode not in {"positive", "negative", "mixed"}:
            raise ValueError(
                "default_heatmap_mode must be 'positive', 'negative', or 'mixed', "
                f"got {default_heatmap_mode!r}"
            )
        self.method = method
        self.target_layer = target_layer
        self.device = torch.device(device) if device is not None else auto_detect_device()
        self.regularization = regularization
        self.flatten_strategy = flatten_strategy
        self.decomposition_space = decomposition_space
        self.default_heatmap_mode = default_heatmap_mode
        self.metadata_: dict[str, Any] | None = None

    def fit(
        self,
        model: torch.nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        max_samples: int | None = 128,
        verbose: bool = False,
        **calibration_kwargs: Any,
    ) -> "EasyVisionExplainer":
        self.model_ = model.to(self.device).eval()
        self.cam_ = NObSPVision(
            method=self.method,
            target_layer=self.target_layer,
            device=self.device,
            regularization=self.regularization,
            flatten_strategy=self.flatten_strategy,
            decomposition_space=self.decomposition_space,
        )
        self.cam_.fit(
            dataloader=calibration_loader,
            model=self.model_,
            max_samples=max_samples,
            verbose=verbose,
            **calibration_kwargs,
        )
        self._sync_fitted_attributes()
        return self

    def save(self, filepath: str, metadata: dict[str, Any] | None = None) -> None:
        self._check_is_fitted()
        self.cam_.save_model(filepath, metadata=metadata)

    def load(self, filepath: str, model: torch.nn.Module) -> "EasyVisionExplainer":
        self.cam_ = NObSPVision(
            method=self.method,
            target_layer=self.target_layer,
            device=self.device,
            regularization=self.regularization,
            flatten_strategy=self.flatten_strategy,
            decomposition_space=self.decomposition_space,
        )
        self.model_ = model.to(self.device).eval()
        self.metadata_ = self.cam_.load_model(filepath, self.model_)
        self._sync_fitted_attributes()
        return self

    def explain(
        self,
        images: torch.Tensor,
        target_classes: int | list[int] | None = None,
        return_features: bool = False,
        heatmap_mode: Literal["positive", "negative", "mixed"] | None = None,
    ) -> list[dict[str, Any]]:
        self._check_is_fitted()
        if images.ndim == 3:
            images = images.unsqueeze(0)
        raw_results = self.cam_.transform(
            images=images.to(self.device),
            model=self.model_,
            target_classes=target_classes,
            return_features=return_features,
        )
        return self._format_results(raw_results, heatmap_mode=heatmap_mode)

    def explain_positive(
        self,
        images: torch.Tensor,
        target_classes: int | list[int] | None = None,
        return_features: bool = False,
    ) -> list[dict[str, Any]]:
        return self.explain(
            images=images,
            target_classes=target_classes,
            return_features=return_features,
            heatmap_mode="positive",
        )

    def explain_negative(
        self,
        images: torch.Tensor,
        target_classes: int | list[int] | None = None,
        return_features: bool = False,
    ) -> list[dict[str, Any]]:
        return self.explain(
            images=images,
            target_classes=target_classes,
            return_features=return_features,
            heatmap_mode="negative",
        )

    def explain_mixed(
        self,
        images: torch.Tensor,
        target_classes: int | list[int] | None = None,
        return_features: bool = False,
    ) -> list[dict[str, Any]]:
        return self.explain(
            images=images,
            target_classes=target_classes,
            return_features=return_features,
            heatmap_mode="mixed",
        )

    def fit_transform(
        self,
        model: torch.nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        images: torch.Tensor,
        target_classes: int | list[int] | None = None,
        max_samples: int | None = 128,
        verbose: bool = False,
        return_features: bool = False,
        heatmap_mode: Literal["positive", "negative", "mixed"] | None = None,
        **calibration_kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.fit(
            model=model,
            calibration_loader=calibration_loader,
            max_samples=max_samples,
            verbose=verbose,
            **calibration_kwargs,
        )
        return self.explain(
            images=images,
            target_classes=target_classes,
            return_features=return_features,
            heatmap_mode=heatmap_mode,
        )

    def feature_importance(self) -> Any:
        self._check_is_fitted()
        return self.cam_.get_feature_importance()

    def _format_results(
        self,
        results: list[dict[str, Any]],
        heatmap_mode: Literal["positive", "negative", "mixed"] | None = None,
    ) -> list[dict[str, Any]]:
        selected_mode = heatmap_mode or self.default_heatmap_mode
        formatted_results: list[dict[str, Any]] = []
        heatmap_key_map = {
            "positive": "heatmap_positive",
            "negative": "heatmap_negative",
            "mixed": "heatmap_mixed",
        }

        for result in results:
            formatted = dict(result)
            formatted["heatmap_mixed"] = result.get("heatmap")
            formatted["heatmap_mode"] = selected_mode

            selected_key = heatmap_key_map[selected_mode]
            if selected_key == "heatmap_mixed":
                selected_heatmap = formatted.get("heatmap_mixed")
            else:
                selected_heatmap = formatted.get(selected_key)
                if selected_heatmap is None:
                    selected_heatmap = formatted.get("heatmap_mixed")

            formatted["heatmap"] = selected_heatmap
            formatted_results.append(formatted)

        return formatted_results

    def _sync_fitted_attributes(self) -> None:
        self.coefficients_ = self.cam_.coefficients_
        self.is_fitted_ = self.cam_.is_fitted_
        self.n_classes_ = self.cam_.n_classes_
        self.selected_classes_ = self.cam_.selected_classes_
        self.calibration_targets_ = self.cam_.calibration_targets_

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "cam_") or not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit(...) before generating heatmaps.")


VisionExplainer = EasyVisionExplainer
