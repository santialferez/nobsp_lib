import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nobsp.core.decompose import decompose_alpha, decompose_basic
from nobsp.core.decompose_cnn import build_forward_model, decompose_alpha_cnn
from nobsp.core.nobsp_cam import NObSPCAM
from nobsp.vision import NObSPVision


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 4)
        self.output = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.tensor(
                    [
                        [0.8, -0.1, 0.3],
                        [0.2, 0.7, -0.4],
                        [-0.5, 0.4, 0.6],
                        [0.1, 0.3, 0.9],
                    ],
                    dtype=torch.float32,
                )
            )
            self.hidden.bias.copy_(torch.tensor([1.5, 0.8, 1.2, 0.4], dtype=torch.float32))
            self.output.weight.copy_(
                torch.tensor([[0.6, -0.3, 0.5, 0.2]], dtype=torch.float32)
            )

    def forward(self, x):
        hidden = F.relu(self.hidden(x))
        prediction = self.output(hidden)
        return prediction, hidden


class AddConstantBlock(nn.Module):
    def __init__(self, constant: float):
        super().__init__()
        self.constant = constant
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.constant)


class TinyResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(AddConstantBlock(0.25))
        self.layer2 = nn.Sequential(AddConstantBlock(0.5))
        self.layer3 = nn.Sequential(
            AddConstantBlock(0.75),
            AddConstantBlock(1.0),
            AddConstantBlock(1.25),
        )
        self.layer4 = nn.Sequential(AddConstantBlock(1.5))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0], [2.0]], dtype=torch.float32))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MiniVisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[0.0]], [[0.5]]],
                        [[[0.0]], [[1.0]], [[-0.5]]],
                    ],
                    dtype=torch.float32,
                )
            )
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -0.5],
                        [0.25, 0.75],
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DictBatchDataset(Dataset):
    def __init__(self):
        self.images = torch.tensor(
            [
                [[[1.0, 0.5], [0.0, 1.0]], [[0.0, 0.5], [1.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]]],
                [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.2, 0.2], [0.2, 0.2]]],
                [[[0.5, 0.5], [0.5, 0.5]], [[1.0, 0.0], [1.0, 0.0]], [[0.1, 0.3], [0.2, 0.4]]],
                [[[1.0, 1.0], [0.5, 0.5]], [[0.5, 0.5], [1.0, 1.0]], [[0.4, 0.2], [0.3, 0.1]]],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "images": self.images[idx],
            "labels": self.labels[idx],
        }


class DummyForwardModel(nn.Module):
    def forward(self, x, return_features=False):
        pooled = torch.flatten(x, 1)
        logits = pooled[:, :2]
        if return_features:
            return logits, pooled[:, :2]
        return logits


class RuntimeRegressionTests(unittest.TestCase):
    def test_alpha_matches_basic_in_centered_feature_space(self):
        model = TinyRegressor().eval()
        X = np.array(
            [
                [1.0, 2.0, 0.5],
                [1.5, 0.0, 2.0],
                [2.0, 1.0, 1.5],
                [0.5, 2.5, 1.0],
                [3.0, 1.5, 0.5],
                [2.5, 2.0, 2.5],
            ],
            dtype=np.float32,
        )

        with torch.no_grad():
            y_pred, _ = model(torch.from_numpy(X))

        _, basic_contributions = decompose_basic(
            X=X,
            y_pred=y_pred,
            model=model,
            problem_type="regression",
            device=torch.device("cpu"),
            regularization=1e-6,
        )
        _, alpha_contributions = decompose_alpha(
            X=X,
            y_pred=y_pred,
            model=model,
            problem_type="regression",
            device=torch.device("cpu"),
            regularization=1e-6,
        )

        np.testing.assert_allclose(alpha_contributions, basic_contributions, atol=1e-5)

    def test_load_model_preserves_none_target_layer(self):
        model = TinyResNetLike().eval()
        fitted = NObSPVision(method="alpha", target_layer=None, device="cpu")
        fitted._nobsp_cam = NObSPCAM(
            model=model,
            target_layer=None,
            method="alpha",
            device="cpu",
        )
        fitted._nobsp_cam.cached_coefficients = np.zeros((1, 2), dtype=np.float32)
        fitted.is_fitted_ = True
        fitted.n_classes_ = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "vision_model.npz"
            fitted.save_model(model_path)

            loaded = NObSPVision(device="cpu")
            metadata = loaded.load_model(model_path, model=model)

        self.assertIsNone(loaded.target_layer)
        self.assertIn("target_layer", metadata)

    def test_calibrate_accepts_dict_batches_without_tensor_boolean_coercion(self):
        model = MiniVisionNet().eval()
        loader = DataLoader(DictBatchDataset(), batch_size=2, shuffle=False)

        cam = NObSPCAM(
            model=model,
            target_layer="conv",
            method="alpha",
            device="cpu",
        )
        cam.calibrate(loader, max_samples=4, verbose=False)

        self.assertIsNotNone(cam.cached_coefficients)
        self.assertEqual(cam.cached_coefficients.shape[1], 4)

    def test_resnet_forward_model_keeps_current_stage_tail(self):
        model = TinyResNetLike().eval()
        x = torch.ones(2, 1, 2, 2)

        with torch.no_grad():
            after_layer2 = model.layer2(model.layer1(x))
            cut_inside_block = model.layer3[1](model.layer3[0](after_layer2))
            expected_inside = model.layer3[2](cut_inside_block)
            expected_inside = model.layer4(expected_inside)
            expected_inside = model.fc(torch.flatten(model.avgpool(expected_inside), 1))

            forward_inside = build_forward_model(model, "layer3.1.relu", torch.device("cpu"))
            actual_inside = forward_inside(cut_inside_block)

            cut_after_stage = model.layer3(after_layer2)
            expected_stage = model.layer4(cut_after_stage)
            expected_stage = model.fc(torch.flatten(model.avgpool(expected_stage), 1))

            forward_stage = build_forward_model(model, "layer3", torch.device("cpu"))
            actual_stage = forward_stage(cut_after_stage)

        np.testing.assert_allclose(actual_inside.numpy(), expected_inside.numpy(), atol=1e-6)
        np.testing.assert_allclose(actual_stage.numpy(), expected_stage.numpy(), atol=1e-6)

    def test_alpha_cnn_failed_solve_zeros_slice_instead_of_crashing(self):
        X = torch.tensor(
            [
                [[[1.0]], [[2.0]]],
                [[[3.0]], [[4.0]]],
                [[[5.0]], [[6.0]]],
            ],
            dtype=torch.float32,
        )
        y_pred = torch.tensor(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=torch.float32,
        )

        with mock.patch(
            "nobsp.core.decompose_cnn.torch.linalg.solve",
            side_effect=RuntimeError("rank deficient"),
        ):
            alpha, contributions = decompose_alpha_cnn(
                X=X,
                y_pred=y_pred,
                forward_model=DummyForwardModel(),
                device=torch.device("cpu"),
                regularization=1e-6,
                pooled_feature_dim=2,
            )

        np.testing.assert_allclose(alpha, 0.0, atol=0.0)
        np.testing.assert_allclose(contributions, 0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
