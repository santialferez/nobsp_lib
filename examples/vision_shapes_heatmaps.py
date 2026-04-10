"""Simple synthetic vision example using NObSP heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from nobsp_lib import EasyVisionExplainer, SmallConvNet, plot_vision_gallery


class ShapesDataset(Dataset):
    """Binary dataset with a bright square on the left or right side."""

    def __init__(self, n_samples: int, image_size: int = 32, noise: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = []
        self.labels = []
        square = image_size // 4
        top = image_size // 2 - square // 2
        bottom = top + square

        for _ in range(n_samples):
            label = int(rng.integers(0, 2))
            image = rng.normal(loc=0.0, scale=noise, size=(image_size, image_size)).astype(np.float32)
            if label == 0:
                left = 3
            else:
                left = image_size - square - 3
            right = left + square
            image[top:bottom, left:right] += 1.5
            image = np.clip(image, 0.0, 1.0)
            self.images.append(image[None, ...])
            self.labels.append(label)

        self.images = torch.tensor(np.stack(self.images), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "vision"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_train = ShapesDataset(n_samples=args.train_samples + args.calibration_samples, seed=0)
    train_size = args.train_samples
    calibration_size = args.calibration_samples
    train_dataset, calibration_dataset = random_split(
        full_train,
        [train_size, calibration_size],
        generator=torch.Generator().manual_seed(0),
    )
    test_dataset = ShapesDataset(n_samples=args.test_samples, seed=1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SmallConvNet(in_channels=1, num_classes=2)
    train_model(model, train_loader, device=device, epochs=args.epochs, lr=1e-3)

    y_true = np.concatenate([labels.numpy() for _, labels in test_loader], axis=0)
    y_pred = predict(model, test_loader, device=device)
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")

    explainer = EasyVisionExplainer(method="beta", device=device)
    explainer.fit(
        model=model,
        calibration_loader=calibration_loader,
        max_samples=args.calibration_samples,
        verbose=False,
    )

    sample_images, _ = next(iter(test_loader))
    sample_images = sample_images[:6]
    results = explainer.explain(sample_images)

    plot_vision_gallery(
        sample_images.cpu().numpy(),
        results,
        class_names=["left square", "right square"],
        save_path=output_dir / "heatmap_gallery.png",
    )

    print(f"Saved vision outputs to {output_dir}")


if __name__ == "__main__":
    main()
