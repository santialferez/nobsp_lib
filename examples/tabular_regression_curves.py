"""Simple synthetic tabular regression example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from nobsp_lib import (
    TabularExplainer,
    plot_importance_heatmap,
    plot_tabular_feature_curves,
    plot_training_history,
)


def make_dataset(n_samples: int = 800, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 4)).astype(np.float32)
    y = (
        np.sin(2.0 * X[:, 0])
        + 0.7 * X[:, 1] ** 2
        - 0.6 * X[:, 2]
        + 0.4 * np.cos(3.0 * X[:, 3])
        + 0.1 * rng.normal(size=n_samples)
    ).astype(np.float32)
    return X, y


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "tabular"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_dataset()
    feature_names = ["sin driver", "quadratic", "linear", "cos driver"]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)

    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    losses = []
    for _ in range(220):
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    explainer = TabularExplainer(task="regression", method="alpha")
    explainer.fit(
        model,
        X,
        feature_names=feature_names,
        transform=scaler.transform,
    )

    y_pred = explainer.predict(X)
    contributions = explainer.explain()
    importance = explainer.feature_importance()

    plot_training_history(
        losses,
        title="Synthetic regression training loss",
        save_path=output_dir / "training_curve.png",
    )
    plot_tabular_feature_curves(
        X,
        contributions,
        feature_names=feature_names,
        save_path=output_dir / "feature_curves.png",
    )
    plot_importance_heatmap(
        importance,
        feature_names=feature_names,
        output_names=["prediction"],
        save_path=output_dir / "importance_heatmap.png",
    )

    print(f"Saved tabular outputs to {output_dir}")
    print(f"Training R2: {r2_score(y, y_pred):.4f}")


if __name__ == "__main__":
    main()
