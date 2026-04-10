"""
GPU-batched β-calibration backend with class batching, logging, and checkpoints.

This module implements a multi-RHS solver for NObSP β-calibration that batches
features and classes on GPU, streams coefficient blocks to CPU (optionally via
memmap checkpoints), and supports resumable execution with structured logging.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from nobsp.core.oblique import oblique_projection_beta

__all__ = ["beta_calibrate_batched"]


LOGGER_NAME = "nobsp.calibration"
METADATA_FILENAME = "metadata.json"
SELECTED_CLASSES_FILENAME = "selected_classes.json"
PROGRESS_STATE_FILENAME = "progress.json"
PROGRESS_LOG_FILENAME = "progress.jsonl"
COEFFICIENTS_FILENAME = "beta_coefficients.npy"
BACKEND_NAME = "gpu_batched_multiclass"


def _utc_iso() -> str:
    """Return current UTC timestamp in ISO8601 Z format."""
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically persist JSON payload to disk."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _ensure_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    if logger is not None:
        return logger
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


@dataclass
class ProgressState:
    next_feature: int = 0
    class_batch_index: int = 0
    features_completed: int = 0
    columns_completed: int = 0
    updated_at: str = field(default_factory=_utc_iso)

    def as_dict(self) -> dict:
        return {
            "next_feature": int(self.next_feature),
            "class_batch_index": int(self.class_batch_index),
            "features_completed": int(self.features_completed),
            "columns_completed": int(self.columns_completed),
            "updated_at": self.updated_at,
        }

    def update_timestamp(self) -> None:
        self.updated_at = _utc_iso()


class CheckpointManager:
    """Handle coefficient storage, metadata, and progress for resumable runs."""

    def __init__(
        self,
        *,
        root: Optional[Path],
        hidden_size: int,
        n_features: int,
        n_selected_classes: int,
        total_classes: int,
        selected_classes: Sequence[int],
        config: dict,
        resume: bool,
        overwrite: bool,
    ) -> None:
        self.enabled = root is not None
        self.root = root
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_selected_classes = n_selected_classes
        self.total_classes = total_classes
        self.selected_classes = list(int(cls) for cls in selected_classes)
        self.resume_requested = resume
        self.overwrite = overwrite
        self.config = dict(config)

        self.metadata_path: Optional[Path] = None
        self.selected_classes_path: Optional[Path] = None
        self.progress_path: Optional[Path] = None
        self.progress_log_path: Optional[Path] = None
        self.coefficients_path: Optional[Path] = None

        self.progress = ProgressState()
        self.metadata: dict = {}

        self._memmap: Optional[np.memmap] = None
        self._tensor_buffer: Optional[torch.Tensor] = None

        if self.enabled:
            self._initialise_paths()
            self._initialise_checkpoint()
        else:
            self.metadata = dict(config)
            self.metadata.update(
                {
                    "backend": BACKEND_NAME,
                    "n_features": n_features,
                    "hidden_size": hidden_size,
                    "n_selected_classes": n_selected_classes,
                    "total_classes": total_classes,
                    "selected_classes": self.selected_classes,
                    "status": "running",
                    "started_at": _utc_iso(),
                }
            )
            self._tensor_buffer = torch.zeros(
                hidden_size, n_selected_classes * n_features, dtype=torch.float32
            )

    def _initialise_paths(self) -> None:
        assert self.root is not None  # guarded by caller
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / METADATA_FILENAME
        self.selected_classes_path = self.root / SELECTED_CLASSES_FILENAME
        self.progress_path = self.root / PROGRESS_STATE_FILENAME
        self.progress_log_path = self.root / PROGRESS_LOG_FILENAME
        self.coefficients_path = self.root / COEFFICIENTS_FILENAME

    def _initialise_checkpoint(self) -> None:
        assert self.metadata_path is not None
        metadata_exists = self.metadata_path.exists()
        resume = self.resume_requested and metadata_exists

        if metadata_exists and not resume and not self.overwrite:
            raise FileExistsError(
                f"Checkpoint metadata exists at {self.metadata_path}. "
                "Pass overwrite=True or resume=True to continue."
            )

        if resume:
            self._load_existing_checkpoint()
        else:
            self._create_fresh_checkpoint()

    def _load_existing_checkpoint(self) -> None:
        assert self.metadata_path is not None
        assert self.coefficients_path is not None
        with self.metadata_path.open("r", encoding="utf-8") as fh:
            stored_metadata = json.load(fh)

        self._validate_metadata(stored_metadata)
        self.metadata = stored_metadata
        self.metadata["resumed_at"] = _utc_iso()
        self.metadata["status"] = "running"

        if self.selected_classes_path and self.selected_classes_path.exists():
            with self.selected_classes_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            stored_selected = payload.get("selected_classes", [])
            if list(stored_selected) != self.selected_classes:
                raise ValueError(
                    "Selected classes in checkpoint do not match current request."
                )

        if self.progress_path and self.progress_path.exists():
            with self.progress_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.progress = ProgressState(
                next_feature=int(payload.get("next_feature", 0)),
                class_batch_index=int(payload.get("class_batch_index", 0)),
                features_completed=int(payload.get("features_completed", 0)),
                columns_completed=int(payload.get("columns_completed", 0)),
                updated_at=payload.get("updated_at", _utc_iso()),
            )
        else:
            self.progress = ProgressState()

        mode = "r+"
        shape = (self.hidden_size, self.n_selected_classes * self.n_features)
        self._memmap = np.memmap(
            self.coefficients_path, dtype=np.float32, mode=mode, shape=shape
        )

    def _create_fresh_checkpoint(self) -> None:
        assert self.metadata_path is not None
        assert self.selected_classes_path is not None
        assert self.coefficients_path is not None

        if self.coefficients_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Coefficient checkpoint exists at {self.coefficients_path}. "
                "Pass overwrite=True to replace it."
            )

        self.metadata = dict(self.config)
        self.metadata.update(
            {
                "backend": BACKEND_NAME,
                "n_features": self.n_features,
                "hidden_size": self.hidden_size,
                "n_selected_classes": self.n_selected_classes,
                "total_classes": self.total_classes,
                "selected_classes": self.selected_classes,
                "status": "running",
                "started_at": _utc_iso(),
            }
        )
        _atomic_write_json(self.metadata_path, self.metadata)

        selected_payload = {
            "total_classes": self.total_classes,
            "selected_classes": self.selected_classes,
        }
        _atomic_write_json(self.selected_classes_path, selected_payload)

        self.progress = ProgressState()
        _atomic_write_json(self.progress_path, self.progress.as_dict())

        shape = (self.hidden_size, self.n_selected_classes * self.n_features)
        self._memmap = np.memmap(
            self.coefficients_path, dtype=np.float32, mode="w+", shape=shape
        )

    def _validate_metadata(self, stored: dict) -> None:
        required = {
            "backend": BACKEND_NAME,
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "n_selected_classes": self.n_selected_classes,
            "total_classes": self.total_classes,
        }
        for key, expected in required.items():
            actual = stored.get(key)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint metadata mismatch for '{key}': expected {expected}, got {actual}"
                )

    def update_metadata(self, **updates: object) -> None:
        self.metadata.update(updates)
        if self.enabled and self.metadata_path is not None:
            _atomic_write_json(self.metadata_path, self.metadata)

    def update_progress(self, **updates: int) -> None:
        for key, value in updates.items():
            setattr(self.progress, key, int(value))
        self.progress.update_timestamp()
        if self.enabled and self.progress_path is not None:
            _atomic_write_json(self.progress_path, self.progress.as_dict())

    def append_progress_log(self, entry: dict) -> None:
        if not self.enabled or self.progress_log_path is None:
            return
        entry_with_ts = dict(entry)
        entry_with_ts.setdefault("ts", _utc_iso())
        with self.progress_log_path.open("a", encoding="utf-8") as fh:
            json.dump(entry_with_ts, fh, separators=(",", ":"))
            fh.write("\n")

    # Coefficient storage ------------------------------------------------------------------
    def write_coefficients(
        self,
        feature_idx: int,
        class_start: int,
        beta_slice: torch.Tensor,
    ) -> None:
        """Persist beta slice for a given feature and class batch."""
        class_count = beta_slice.shape[1]
        target_cols = [
            (class_start + offset) * self.n_features + feature_idx
            for offset in range(class_count)
        ]
        beta_cpu = beta_slice.detach().cpu().to(torch.float32)
        if self._memmap is not None:
            self._memmap[:, target_cols] = beta_cpu.numpy()
        else:
            if self._tensor_buffer is None:
                self._tensor_buffer = torch.zeros(
                    self.hidden_size,
                    self.n_selected_classes * self.n_features,
                    dtype=torch.float32,
                )
            self._tensor_buffer[:, target_cols] = beta_cpu

    def flush(self) -> None:
        if self._memmap is not None:
            self._memmap.flush()

    def coefficients(self) -> torch.Tensor:
        if self._memmap is not None:
            self._memmap.flush()
            array = np.array(self._memmap, copy=True)
            return torch.from_numpy(array)
        if self._tensor_buffer is None:
            return torch.zeros(
                self.hidden_size,
                self.n_selected_classes * self.n_features,
                dtype=torch.float32,
            )
        return self._tensor_buffer.clone()

    def close(self, status: str, wall_time: float) -> None:
        self.update_metadata(
            status=status,
            completed_at=_utc_iso(),
            wall_time_seconds=float(wall_time),
            features_completed=int(self.progress.features_completed),
            columns_completed=int(self.progress.columns_completed),
        )
        self.flush()


def _prepare_class_batches(
    n_selected_classes: int, class_batch_size: Optional[int]
) -> List[Tuple[int, int]]:
    if class_batch_size is None or class_batch_size <= 0:
        class_batch_size = n_selected_classes
    class_batch_size = min(class_batch_size, n_selected_classes)
    if class_batch_size <= 0:
        class_batch_size = n_selected_classes
    batches: List[Tuple[int, int]] = []
    for start in range(0, n_selected_classes, class_batch_size):
        end = min(n_selected_classes, start + class_batch_size)
        batches.append((start, end))
    return batches


def _resolve_device(
    device: Optional[torch.device], tensors: Iterable[torch.Tensor], model: torch.nn.Module
) -> torch.device:
    if device is not None:
        return torch.device(device)
    if any(t.is_cuda for t in tensors):
        return torch.device(tensors[0].device)
    try:
        param = next(model.parameters())
        return param.device
    except StopIteration:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def beta_calibrate_batched(
    X: torch.Tensor,
    Y: torch.Tensor,
    model: torch.nn.Module,
    *,
    lambda_reg: float = 1e-6,
    device: Optional[torch.device] = None,
    feature_batch_size: int = 16,
    class_batch_size: Optional[int] = None,
    coefficients_only: bool = True,  # noqa: ARG001 - retained for API parity
    use_mixed_precision: bool = True,
    solver: str = "chol",
    decomposition_space: str = "classifier_input",
    selected_classes: Optional[Sequence[int]] = None,
    checkpoint_path: Optional[os.PathLike[str] | str] = None,
    checkpoint_interval_features: int = 16,
    checkpoint_interval_minutes: int = 5,
    resume: bool = True,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """
    Compute β-calibration coefficients using GPU batched multi-RHS solves.
    """

    log = _ensure_logger(logger)

    if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
        raise TypeError("X and Y must be torch.Tensor instances.")
    if X.dim() != 2:
        raise ValueError(f"Expected X with shape (n_samples, n_features); got {tuple(X.shape)}.")
    if Y.dim() != 2:
        raise ValueError(f"Expected Y with shape (n_samples, n_classes); got {tuple(Y.shape)}.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must share the same number of samples.")
    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be a positive integer.")
    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be positive.")

    n_samples, n_features = X.shape
    _, total_classes = Y.shape

    device = _resolve_device(device, (X, Y), model)
    model = model.to(device).eval()

    X_device = X.to(device=device, dtype=torch.float32, non_blocking=True)
    Y_device = Y.to(device=device, dtype=torch.float32, non_blocking=True)

    if selected_classes is None:
        class_ids = list(range(total_classes))
    else:
        if isinstance(selected_classes, torch.Tensor):
            class_ids = [int(c) for c in selected_classes.tolist()]
        else:
            class_ids = [int(c) for c in selected_classes]
        invalid = [c for c in class_ids if c < 0 or c >= total_classes]
        if invalid:
            raise ValueError(f"selected_classes contains invalid indices: {invalid}")
    n_selected_classes = len(class_ids)
    if n_selected_classes == 0:
        raise ValueError("At least one class must be selected for calibration.")

    class_batches = _prepare_class_batches(n_selected_classes, class_batch_size)

    solver_lower = (solver or "").lower()
    multi_solver_aliases = {"chol_multi", "multi_chol", "chol_reuse", "multi"}
    multi_rhs_requested = solver_lower in multi_solver_aliases
    multi_rhs_features_success = 0
    multi_rhs_features_fallback = 0
    multi_rhs_first_fallback_logged = False

    class_index_tensor = torch.tensor(class_ids, device=device, dtype=torch.long)
    Y_selected = torch.index_select(Y_device, dim=1, index=class_index_tensor)

    with torch.no_grad():
        sample_input = X_device[:1]
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=amp_dtype)
            if device.type == "cuda" and use_mixed_precision
            else nullcontext()
        )
        with autocast_ctx:
            _, hidden_probe, _ = model(sample_input)
    hidden_size = hidden_probe.shape[-1]

    checkpoint_root = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint = CheckpointManager(
        root=checkpoint_root,
        hidden_size=hidden_size,
        n_features=n_features,
        n_selected_classes=n_selected_classes,
        total_classes=total_classes,
        selected_classes=class_ids,
        config={
            "lambda_reg": float(lambda_reg),
            "solver": solver,
            "feature_batch_size": int(feature_batch_size),
            "class_batch_size": int(class_batch_size) if class_batch_size else None,
            "device": str(device),
            "decomposition_space": decomposition_space,
            "use_mixed_precision": bool(use_mixed_precision),
            "coefficients_only": bool(coefficients_only),
        },
        resume=resume,
        overwrite=overwrite,
    )

    progress = checkpoint.progress
    features_completed_total = progress.features_completed
    start_feature = progress.next_feature
    resumed_class_batch_index = progress.class_batch_index

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_batches = math.ceil((n_features - start_feature) / feature_batch_size) if n_features > start_feature else 0
    log.info(
        "Starting β-calibration (%s): device=%s solver=%s dtype=%s feature_batch=%d class_batch=%s "
        "selected_classes=%d/%d decomposition_space=%s samples=%d features=%d hidden=%d λ=%.2e "
        "resume=%s total_batches=%d",
        BACKEND_NAME,
        device,
        solver,
        "mixed" if (device.type == "cuda" and use_mixed_precision) else "fp32",
        feature_batch_size,
        class_batch_size if class_batch_size else "all",
        n_selected_classes,
        total_classes,
        decomposition_space,
        n_samples,
        n_features,
        hidden_size,
        lambda_reg,
        resume and start_feature > 0,
        total_batches,
    )

    start_time = time.perf_counter()
    last_checkpoint_time = start_time
    features_since_checkpoint = 0
    checkpoint_interval_seconds = max(1, int(checkpoint_interval_minutes * 60))

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=amp_dtype)
        if device.type == "cuda" and use_mixed_precision
        else nullcontext()
    )

    try:
        while start_feature < n_features:
            batch_end = min(n_features, start_feature + feature_batch_size)
            batch_features = list(range(start_feature, batch_end))
            batch_size = len(batch_features)

            # Build flattened inputs for target and reference
            input_dtype = (
                amp_dtype if (device.type == "cuda" and use_mixed_precision) else X_device.dtype
            )
            X_target_batch = torch.zeros(
                (batch_size, n_samples, n_features),
                device=device,
                dtype=input_dtype,
            )
            X_reference_batch = (
                X_device.to(device=device, dtype=input_dtype)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
                .clone()
            )

            for offset, feature_idx in enumerate(batch_features):
                feature_column = X_device[:, feature_idx].to(device=device, dtype=input_dtype)
                X_target_batch[offset, :, feature_idx] = feature_column
                X_reference_batch[offset, :, feature_idx] = 0.0

            X_target_flat = X_target_batch.view(batch_size * n_samples, n_features)
            X_reference_flat = X_reference_batch.view(batch_size * n_samples, n_features)

            with torch.no_grad():
                with autocast_ctx:
                    _, hidden_target_flat, _ = model(X_target_flat)
                    _, hidden_reference_flat, _ = model(X_reference_flat)

            hidden_target = hidden_target_flat.view(batch_size, n_samples, hidden_size).float()
            hidden_reference = hidden_reference_flat.view(batch_size, n_samples, hidden_size).float()

            del X_target_flat, X_reference_flat, hidden_target_flat, hidden_reference_flat

            for local_idx, feature_idx in enumerate(batch_features):
                class_start_idx = 0
                if feature_idx == progress.next_feature:
                    class_start_idx = resumed_class_batch_index
                elif feature_idx < progress.next_feature:
                    continue  # Already completed via resume bookkeeping

                if class_start_idx >= len(class_batches):
                    # This feature already completed previously.
                    continue

                X_k = hidden_target[local_idx]
                X_notk = hidden_reference[local_idx]

                X_kc = (X_k - X_k.mean(dim=0)).clone()
                X_notkc = (X_notk - X_notk.mean(dim=0)).clone()

                use_multi_rhs = multi_rhs_requested
                multi_available = False
                chol_X = chol_AtA = None
                A_k = None

                if use_multi_rhs:
                    XtX = X_notkc.T @ X_notkc
                    I_hidden = torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
                    XtX_reg = XtX + float(lambda_reg) * I_hidden
                    try:
                        chol_X = torch.linalg.cholesky(XtX_reg)
                        alpha_x = torch.cholesky_solve((X_notkc.T @ X_kc), chol_X)
                        A_k = X_kc - X_notkc @ alpha_x
                        AtA = A_k.T @ A_k
                        AtA_reg = AtA + float(lambda_reg) * I_hidden
                        chol_AtA = torch.linalg.cholesky(AtA_reg)
                        multi_available = True
                    except (RuntimeError, torch.linalg.LinAlgError) as err:
                        if not multi_rhs_first_fallback_logged:
                            log.warning(
                                "Multi-RHS setup failed for feature %d; falling back to legacy solver (%s)",
                                feature_idx,
                                err,
                            )
                            multi_rhs_first_fallback_logged = True
                        multi_available = False
                    except Exception as err:  # pragma: no cover - defensive
                        if not multi_rhs_first_fallback_logged:
                            log.warning(
                                "Multi-RHS setup failed for feature %d; falling back to legacy solver (%s)",
                                feature_idx,
                                err,
                            )
                            multi_rhs_first_fallback_logged = True
                        multi_available = False

                total_class_batches = len(class_batches)
                for class_batch_pos, (cls_start, cls_end) in enumerate(class_batches):
                    if class_batch_pos < class_start_idx:
                        continue  # Skip completed slices when resuming

                    slice_betas: List[torch.Tensor] = []
                    for class_idx in range(cls_start, cls_end):
                        y_col = Y_selected[:, class_idx]

                        if multi_available:
                            try:
                                rhs_y = (X_notkc.T @ y_col).unsqueeze(1)
                                alpha_y = torch.cholesky_solve(rhs_y, chol_X).squeeze(1)
                                y_k = y_col - X_notkc @ alpha_y
                                rhs = (A_k.T @ y_k).unsqueeze(1)
                                beta_col = torch.cholesky_solve(rhs, chol_AtA).squeeze(1)
                            except (RuntimeError, torch.linalg.LinAlgError) as err:
                                multi_available = False
                                if not multi_rhs_first_fallback_logged:
                                    log.warning(
                                        "Multi-RHS solve fell back at feature %d class %d (%s)",
                                        feature_idx,
                                        class_idx,
                                        err,
                                    )
                                    multi_rhs_first_fallback_logged = True
                                y_centered = y_col - y_col.mean()
                                beta_col = oblique_projection_beta(
                                    X_kc,
                                    X_notkc,
                                    y_centered,
                                    device=device,
                                    lambda_reg=lambda_reg,
                                )
                            except Exception as err:  # pragma: no cover - defensive
                                multi_available = False
                                if not multi_rhs_first_fallback_logged:
                                    log.warning(
                                        "Multi-RHS solve fell back at feature %d class %d (%s)",
                                        feature_idx,
                                        class_idx,
                                        err,
                                    )
                                    multi_rhs_first_fallback_logged = True
                                y_centered = y_col - y_col.mean()
                                beta_col = oblique_projection_beta(
                                    X_kc,
                                    X_notkc,
                                    y_centered,
                                    device=device,
                                    lambda_reg=lambda_reg,
                                )
                        else:
                            y_centered = y_col - y_col.mean()
                            beta_col = oblique_projection_beta(
                                X_kc,
                                X_notkc,
                                y_centered,
                                device=device,
                                lambda_reg=lambda_reg,
                            )
                        slice_betas.append(beta_col.unsqueeze(1))

                    beta_slice = (
                        torch.cat(slice_betas, dim=1)
                        if slice_betas
                        else torch.empty(hidden_size, 0, device=device, dtype=X_kc.dtype)
                    )

                    checkpoint.write_coefficients(feature_idx, cls_start, beta_slice)
                    checkpoint.progress.columns_completed += (cls_end - cls_start)

                    elapsed = time.perf_counter() - start_time
                    partial_feature = (class_batch_pos + 1) / total_class_batches
                    features_fraction = features_completed_total + partial_feature
                    rate = features_fraction / elapsed if elapsed > 0 else 0.0
                    remaining = max(0.0, n_features - features_fraction)
                    eta = remaining / rate if rate > 0 else None
                    peak_vram = (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    )

                    checkpoint.append_progress_log(
                        {
                            "backend": BACKEND_NAME,
                            "device": str(device),
                            "feat_done": features_completed_total,
                            "feat_total": n_features,
                            "cls_total": n_selected_classes,
                            "feat_batch": [feature_idx, feature_idx + 1],
                            "cls_batch": [cls_start, cls_end],
                            "elapsed_s": elapsed,
                            "eta_s": eta,
                            "features_per_s": rate,
                            "peak_vram_bytes": peak_vram,
                            "checkpoint_path": str(checkpoint.coefficients_path)
                            if checkpoint.coefficients_path
                            else None,
                        }
                    )

                    # Update progress state for potential resume
                    next_feature_idx = feature_idx
                    next_class_batch_idx = class_batch_pos + 1
                    if next_class_batch_idx >= total_class_batches:
                        features_completed_total += 1
                        features_since_checkpoint += 1
                        next_feature_idx = feature_idx + 1
                        next_class_batch_idx = 0
                        progress.features_completed = features_completed_total
                    progress.next_feature = next_feature_idx
                    progress.class_batch_index = next_class_batch_idx
                    checkpoint.update_progress(
                        next_feature=progress.next_feature,
                        class_batch_index=progress.class_batch_index,
                        features_completed=progress.features_completed,
                        columns_completed=progress.columns_completed,
                    )

                    now = time.perf_counter()
                    time_since_checkpoint = now - last_checkpoint_time
                    if (
                        features_since_checkpoint >= checkpoint_interval_features
                        or time_since_checkpoint >= checkpoint_interval_seconds
                    ):
                        checkpoint.flush()
                        last_checkpoint_time = now
                        features_since_checkpoint = 0

                # Reset resume state after processing this feature
                resumed_class_batch_index = 0

                if use_multi_rhs:
                    if multi_available:
                        multi_rhs_features_success += 1
                    else:
                        multi_rhs_features_fallback += 1

            start_feature = progress.next_feature

    except Exception:
        wall_time = time.perf_counter() - start_time
        checkpoint.close(status="error", wall_time=wall_time)
        raise

    wall_time = time.perf_counter() - start_time
    checkpoint.close(status="completed", wall_time=wall_time)

    coefficients = checkpoint.coefficients()
    if multi_rhs_requested:
        total_features_tracked = multi_rhs_features_success + multi_rhs_features_fallback
        if total_features_tracked > 0:
            if multi_rhs_features_fallback > 0:
                log.warning(
                    "Multi-RHS solver fell back to legacy for %d/%d features (λ=%.3e)",
                    multi_rhs_features_fallback,
                    total_features_tracked,
                    lambda_reg,
                )
            else:
                log.info(
                    "Multi-RHS solver succeeded for all %d features (λ=%.3e)",
                    total_features_tracked,
                    lambda_reg,
                )
    log.info(
        "β-calibration complete: shape=%s wall_time=%.2fs features=%d classes=%d backend=%s",
        tuple(coefficients.shape),
        wall_time,
        n_features,
        n_selected_classes,
        BACKEND_NAME,
    )
    return coefficients
