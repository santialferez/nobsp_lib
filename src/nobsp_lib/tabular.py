"""Community-facing tabular API built on top of the NObSP core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nobsp import NObSP
from nobsp.utils.device import auto_detect_device

from .adapters import HookedTabularAdapter
from .models import TabularClassifierNet, TabularRegressorNet


@dataclass
class TabularHistory:
    losses: list[float]


class TabularExplainer:
    """Explain an existing PyTorch tabular model with NObSP."""

    def __init__(
        self,
        task: str = "regression",
        method: str = "alpha",
        regularization: float = 1e-4,
        device: str | torch.device | None = None,
    ):
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'.")
        if method not in {"basic", "alpha", "beta"}:
            raise ValueError("method must be 'basic', 'alpha', or 'beta'.")

        self.task = task
        self.method = method
        self.regularization = regularization
        self.device = torch.device(device) if device is not None else auto_detect_device()

    def fit(
        self,
        model: nn.Module,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        *,
        class_names: list[str] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        feature_layer: str | nn.Module | None = None,
        capture: str = "auto",
        output_kind: str = "auto",
        output_selector: int | str | Callable[[Any], Any] | None = None,
    ) -> "TabularExplainer":
        X = self._validate_X(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        self.feature_names_ = feature_names or [f"x{i}" for i in range(X.shape[1])]
        self.class_names_ = class_names
        self.transform_ = transform
        self.model_ = model.to(self.device).eval()
        self.adapter_ = HookedTabularAdapter(
            self.model_,
            task=self.task,
            feature_layer=feature_layer,
            capture=capture,
            output_kind=output_kind,
            output_selector=output_selector,
        ).to(self.device).eval()
        self.X_train_ = self._apply_transform(X)

        self.engine_ = NObSP(
            method=self.method,
            regularization=self.regularization,
            device=self.device,
        )
        self.engine_.fit(self.X_train_, self.adapter_)
        self.train_contributions_ = self.engine_.contributions_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X_tensor = torch.as_tensor(self._apply_transform(X), device=self.device)
        self.adapter_.eval()
        with torch.no_grad():
            outputs = self.adapter_(X_tensor)
        if self.task == "classification":
            probabilities, _, _ = outputs
            indices = probabilities.argmax(dim=1).cpu().numpy()
            if getattr(self, "class_names_", None) is not None:
                class_names = np.asarray(self.class_names_)
                return class_names[indices]
            return indices
        predictions, _ = outputs
        return predictions.squeeze(-1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        X_tensor = torch.as_tensor(self._apply_transform(X), device=self.device)
        self.adapter_.eval()
        with torch.no_grad():
            probabilities, _, _ = self.adapter_(X_tensor)
        return probabilities.cpu().numpy()

    def explain(self, X: np.ndarray | None = None) -> np.ndarray:
        self._check_is_fitted()
        if X is None:
            return self.train_contributions_
        X_transformed = self._apply_transform(X)
        return self.engine_.transform(X_transformed, model=self.adapter_)

    def feature_importance(self) -> np.ndarray:
        self._check_is_fitted()
        return self.engine_.get_feature_importance()

    def save(self, filepath: str | Path) -> None:
        self._check_is_fitted()
        payload = {
            "method": self.method,
            "task": self.task,
            "regularization": self.regularization,
            "components": self.engine_.components_,
            "contributions": self.train_contributions_,
            "n_features_in": self.engine_.n_features_in_,
            "n_outputs": self.engine_.n_outputs_,
            "hidden_size": self.engine_.hidden_size_,
            "problem_type": self.engine_.problem_type_,
            "feature_names": np.asarray(self.feature_names_, dtype=object),
        }
        if self.class_names_ is not None:
            payload["class_names"] = np.asarray(self.class_names_, dtype=object)
        np.savez_compressed(filepath, **payload)

    def load(
        self,
        filepath: str | Path,
        model: nn.Module,
        *,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        feature_layer: str | nn.Module | None = None,
        capture: str = "auto",
        output_kind: str = "auto",
        output_selector: int | str | Callable[[Any], Any] | None = None,
    ) -> "TabularExplainer":
        data = np.load(filepath, allow_pickle=True)
        self.method = str(data["method"])
        self.task = str(data["task"])
        self.regularization = float(data["regularization"])
        self.feature_names_ = data["feature_names"].tolist()
        self.class_names_ = data["class_names"].tolist() if "class_names" in data.files else None
        self.transform_ = transform
        self.model_ = model.to(self.device).eval()
        self.adapter_ = HookedTabularAdapter(
            self.model_,
            task=self.task,
            feature_layer=feature_layer,
            capture=capture,
            output_kind=output_kind,
            output_selector=output_selector,
        ).to(self.device).eval()

        self.engine_ = NObSP(
            method=self.method,
            regularization=self.regularization,
            device=self.device,
        )
        self.engine_.components_ = data["components"]
        self.engine_.contributions_ = data["contributions"]
        self.engine_.n_features_in_ = int(data["n_features_in"])
        self.engine_.n_outputs_ = int(data["n_outputs"])
        self.engine_.hidden_size_ = int(data["hidden_size"])
        self.engine_.problem_type_ = str(data["problem_type"])
        self.engine_._device = self.device
        self.train_contributions_ = self.engine_.contributions_
        return self

    def _apply_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_X(X)
        if getattr(self, "transform_", None) is not None:
            X = np.asarray(self.transform_(X), dtype=np.float32)
        return X.astype(np.float32)

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "engine_"):
            raise RuntimeError("Call fit(...) or load(...) before using the explainer.")


class EasyTabularExplainer(TabularExplainer):
    """Convenience trainer for a default MLP, on top of the general tabular API."""

    def __init__(
        self,
        task: str = "regression",
        method: str = "alpha",
        hidden_dims: Iterable[int] = (64, 32),
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        regularization: float = 1e-4,
        device: str | torch.device | None = None,
        random_state: int = 0,
        use_standard_scaler: bool = True,
    ):
        super().__init__(
            task=task,
            method=method,
            regularization=regularization,
            device=device,
        )
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.use_standard_scaler = use_standard_scaler

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        *,
        class_names: list[str] | None = None,
    ) -> "EasyTabularExplainer":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = self._validate_X(X)
        y = np.asarray(y)

        if self.use_standard_scaler:
            self.scaler_ = StandardScaler().fit(X)
            self.transform_ = self.scaler_.transform
        else:
            self.transform_ = None

        X_train = self._apply_transform(X)
        self.model_ = self._build_model(X_train.shape[1], y, class_names)
        self.history_ = self._train_model(X_train, y)
        super().fit(
            self.model_,
            X,
            feature_names=feature_names,
            class_names=self.class_names_,
            transform=self.transform_,
        )
        return self

    def _build_model(
        self,
        input_dim: int,
        y: np.ndarray,
        class_names: list[str] | None,
    ) -> nn.Module:
        if self.task == "classification":
            classes = np.unique(y)
            self.class_names_ = class_names or classes.tolist()
            self.classes_ = np.asarray(classes)
            self.num_classes_ = len(classes)
            model = TabularClassifierNet(
                input_dim=input_dim,
                num_classes=self.num_classes_,
                hidden_dims=self.hidden_dims,
            )
        else:
            self.class_names_ = None
            model = TabularRegressorNet(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
            )
        return model.to(self.device)

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> TabularHistory:
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        if self.task == "classification":
            y_indices = np.searchsorted(self.classes_, y)
            y_tensor = torch.as_tensor(y_indices, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        else:
            y_tensor = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
            criterion = nn.MSELoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        losses: list[float] = []

        for _ in range(self.epochs):
            self.model_.train()
            running_loss = 0.0
            sample_count = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model_(X_batch)
                if self.task == "classification":
                    _, _, logits = outputs
                    loss = criterion(logits, y_batch)
                else:
                    predictions, _ = outputs
                    loss = criterion(predictions, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = X_batch.shape[0]
                running_loss += loss.item() * batch_size
                sample_count += batch_size

            losses.append(running_loss / max(sample_count, 1))

        return TabularHistory(losses=losses)
