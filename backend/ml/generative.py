"""Small, CPU-only VAE and conditional GAN experiments for real CFD patches."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from backend.services.generative_data import (
    GRID_SIZE,
    PATCH_SIZE,
    PATCH_STRIDE,
    split_patch_origins,
)


MODEL_ID_PATTERN = re.compile(r"^(vae|gan)_[a-z0-9_-]{3,110}$")
LATENT_DIMENSION = 16
HIDDEN_DIMENSION = 256
DEFAULT_EPOCHS = 30
MAX_EPOCHS = 100
MINIMUM_EPOCHS = 1
VAE_BETA = 0.001


class GenerativeError(Exception):
    """A generative model, checkpoint, or input cannot be used safely."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise GenerativeError(
                "PyTorch is not installed in the backend environment. Install the CPU-only "
                "dependency with `python -m pip install -r requirements-generative.txt`.",
                503,
            ) from error
        raise GenerativeError(
            "PyTorch could not initialize because a required Python module is missing.",
            503,
        ) from error
    except (ImportError, OSError) as error:
        raise GenerativeError(
            "PyTorch could not initialize in the backend environment. Check the installed "
            "PyTorch build and its native runtime dependencies.",
            503,
        ) from error
    return torch


def torch_runtime_status() -> dict[str, Any]:
    try:
        torch = _import_torch()
    except GenerativeError as error:
        return {
            "available": False,
            "version": None,
            "device": None,
            "diagnostic": str(error),
        }
    return {
        "available": True,
        "version": str(torch.__version__),
        "device": "cpu",
        "diagnostic": "CPU PyTorch runtime imported successfully.",
    }


def torch_available() -> bool:
    return torch_runtime_status()["available"]


def require_torch():
    return _import_torch()


def validate_training_config(model_type: str, seed: Any, epochs: Any) -> tuple[str, int, int]:
    if not isinstance(model_type, str) or model_type not in {"vae", "gan"}:
        raise GenerativeError("model_type must be either 'vae' or 'gan'.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise GenerativeError("seed must be an integer from 0 through 4294967295.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not MINIMUM_EPOCHS <= epochs <= MAX_EPOCHS:
        raise GenerativeError(f"epochs must be an integer from {MINIMUM_EPOCHS} through {MAX_EPOCHS}.")
    return model_type, seed, epochs


def _patch_arrays(
    dataset: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        values = np.asarray(dataset["values"], dtype=np.float32)
        masks = np.asarray(dataset["masks"], dtype=bool)
        splits = np.asarray(dataset["split_labels"], dtype=str)
        origins = np.asarray(dataset["origins"], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as error:
        raise GenerativeError("Prepared real-data patches are missing required arrays.") from error
    if (
        values.ndim != 4
        or masks.shape != values.shape
        or values.shape[1:] != (8, 8, 8)
        or splits.shape != (values.shape[0],)
        or origins.shape != (values.shape[0], 3)
        or not np.isfinite(values[masks]).all()
    ):
        raise GenerativeError("Prepared real-data patches have an invalid field, mask, or split shape.")
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    if train.size < 32 or validation.size < 8:
        raise GenerativeError("The spatially separated patch sample is too small for this experiment.")
    if int(origins[train, 0].max()) + PATCH_SIZE >= int(origins[validation, 0].min()):
        raise GenerativeError("Training and validation patches overlap or are not spatially buffered.")
    if not np.array_equal(splits, split_patch_origins(origins)):
        raise GenerativeError("Prepared patch split labels do not match their spatial locations.")
    return values, masks, splits


def _selected_indexes(splits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.flatnonzero(splits == "train"), np.flatnonzero(splits == "validation")


def _new_networks(torch):
    nn = torch.nn

    class VariationalAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(1024, HIDDEN_DIMENSION),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION // 2),
                nn.ReLU(),
            )
            self.mean = nn.Linear(HIDDEN_DIMENSION // 2, LATENT_DIMENSION)
            self.log_variance = nn.Linear(HIDDEN_DIMENSION // 2, LATENT_DIMENSION)
            self.decoder = nn.Sequential(
                nn.Linear(LATENT_DIMENSION, HIDDEN_DIMENSION // 2),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIMENSION // 2, HIDDEN_DIMENSION),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIMENSION, 512),
            )

        def forward(self, values, masks, *, sample=True):
            hidden = self.encoder(torch.cat((values * masks, masks), dim=1))
            mean = self.mean(hidden)
            log_variance = self.log_variance(hidden).clamp(-10, 10)
            if sample:
                latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
            else:
                latent = mean
            return self.decoder(latent), mean, log_variance

    class ConditionalGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(LATENT_DIMENSION + 512, HIDDEN_DIMENSION),
                nn.LeakyReLU(0.2),
                nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION),
                nn.LeakyReLU(0.2),
                nn.Linear(HIDDEN_DIMENSION, 512),
            )

        def forward(self, noise, masks):
            return self.network(torch.cat((noise, masks), dim=1)) * masks

    class ConditionalDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(1024, HIDDEN_DIMENSION),
                nn.LeakyReLU(0.2),
                nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(HIDDEN_DIMENSION // 2, 1),
            )

        def forward(self, values, masks):
            return self.network(torch.cat((values * masks, masks), dim=1))

    return VariationalAutoencoder, ConditionalGenerator, ConditionalDiscriminator


def _normalization(values: np.ndarray, masks: np.ndarray) -> tuple[float, float]:
    valid = values[masks].astype(np.float64)
    mean = float(valid.mean())
    standard_deviation = float(valid.std())
    if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation < 1e-8:
        raise GenerativeError("Training patches do not contain a finite varying scalar field.")
    return mean, standard_deviation


def _to_tensor(torch, values: np.ndarray, masks: np.ndarray, mean: float, scale: float):
    normalized = np.where(masks, (values - mean) / scale, 0).astype(np.float32)
    return (
        torch.as_tensor(normalized.reshape(-1, 512), dtype=torch.float32),
        torch.as_tensor(masks.reshape(-1, 512).astype(np.float32), dtype=torch.float32),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _save_checkpoint(torch, path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _model_paths(models_root: Path, model_id: str) -> tuple[Path, Path]:
    if not isinstance(model_id, str) or not MODEL_ID_PATTERN.fullmatch(model_id):
        raise GenerativeError("The requested generative model ID is invalid.")
    directory = models_root / "generative" / "models" / model_id
    return directory / "metadata.json", directory / "checkpoint.pt"


def _model_id(
    model_type: str, case_id: str, manifest_sha256: str, seed: int, epochs: int
) -> str:
    case_slug = re.sub(r"[^a-z0-9]+", "_", case_id.lower()).strip("_")
    suffix = hashlib.sha256(
        f"{case_id}:{manifest_sha256}:{seed}:{epochs}:{GRID_SIZE}:{PATCH_SIZE}:{PATCH_STRIDE}".encode()
    ).hexdigest()[:10]
    return f"{model_type}_{case_slug}_{suffix}"


def reconstruction_metrics(
    actual: Any, reconstructed: Any, masks: Any
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=np.float64)
    reconstructed_array = np.asarray(reconstructed, dtype=np.float64)
    mask_array = np.asarray(masks, dtype=bool)
    if (
        actual_array.shape != reconstructed_array.shape
        or mask_array.shape != actual_array.shape
        or not mask_array.any()
        or not np.isfinite(actual_array[mask_array]).all()
        or not np.isfinite(reconstructed_array[mask_array]).all()
    ):
        raise GenerativeError("Reconstruction metrics require aligned finite values and a non-empty mask.")
    difference = reconstructed_array[mask_array] - actual_array[mask_array]
    return {
        "mae_K": float(np.mean(np.abs(difference))),
        "rmse_K": float(np.sqrt(np.mean(np.square(difference)))),
    }


def positive_temperature_check(values: Any, masks: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    validity = np.asarray(masks, dtype=bool)
    if array.shape != validity.shape or not validity.any() or not np.isfinite(array[validity]).all():
        raise GenerativeError("The absolute-temperature check requires finite Kelvin values and a valid mask.")
    fraction = float(np.mean(array[validity] > 0))
    return {
        "positive_kelvin_fraction": fraction,
        "positivity_passed": bool(np.all(array[validity] > 0)),
        "physical_plausibility_check": "absolute temperature greater than 0 K",
        "plausibility_caveat": "Positivity is a minimal check, not physical validation.",
    }


def _neighbor_jump_mean(values: np.ndarray, masks: np.ndarray) -> float | None:
    jumps = []
    for axis in range(1, 4):
        left_slice = [slice(None)] * 4
        right_slice = [slice(None)] * 4
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)
        left = tuple(left_slice)
        right = tuple(right_slice)
        adjacent = masks[left] & masks[right]
        if adjacent.any():
            jumps.extend(np.abs(values[left][adjacent] - values[right][adjacent]).tolist())
    return float(np.mean(jumps)) if jumps else None


def evaluate_generated_samples(
    generated: Any, generated_masks: Any, heldout: Any, heldout_masks: Any
) -> dict[str, Any]:
    generated_values = np.asarray(generated, dtype=np.float64)
    generated_validity = np.asarray(generated_masks, dtype=bool)
    heldout_values = np.asarray(heldout, dtype=np.float64)
    heldout_validity = np.asarray(heldout_masks, dtype=bool)
    if (
        generated_values.shape != generated_validity.shape
        or heldout_values.shape != heldout_validity.shape
        or generated_values.ndim != 4
        or heldout_values.ndim != 4
        or not generated_validity.any()
        or not heldout_validity.any()
    ):
        raise GenerativeError("GAN evaluation requires aligned generated and held-out real patches.")
    generated_flat = generated_values[generated_validity]
    heldout_flat = heldout_values[heldout_validity]
    if not np.isfinite(generated_flat).all() or not np.isfinite(heldout_flat).all():
        raise GenerativeError("GAN evaluation cannot include non-finite valid field values.")
    generated_jumps = _neighbor_jump_mean(generated_values, generated_validity)
    heldout_jumps = _neighbor_jump_mean(heldout_values, heldout_validity)
    positivity = positive_temperature_check(generated_values, generated_validity)
    return {
        "heldout_real_voxel_count": int(heldout_flat.size),
        "generated_valid_voxel_count": int(generated_flat.size),
        "temperature_wasserstein_distance_K": float(
            wasserstein_distance(heldout_flat, generated_flat)
        ),
        "heldout_mean_temperature_K": float(heldout_flat.mean()),
        "generated_mean_temperature_K": float(generated_flat.mean()),
        "heldout_temperature_std_K": float(heldout_flat.std()),
        "generated_temperature_std_K": float(generated_flat.std()),
        "heldout_mean_neighbor_jump_K_per_voxel": heldout_jumps,
        "generated_mean_neighbor_jump_K_per_voxel": generated_jumps,
        "mean_neighbor_jump_difference_K_per_voxel": (
            abs(heldout_jumps - generated_jumps)
            if heldout_jumps is not None and generated_jumps is not None
            else None
        ),
        **positivity,
        "physical_plausibility": {
            "check": "finite values and absolute temperature greater than 0 K",
            "all_valid_values_finite": True,
            "positive_kelvin_fraction": positivity["positive_kelvin_fraction"],
            "positivity_passed": positivity["positivity_passed"],
            "caveat": "Positivity and statistical similarity do not establish physical validity or CFD convergence.",
        },
        "status": "exploratory_distribution_comparison",
    }


def generated_sample_metadata(sample: dict[str, Any], seed: int) -> dict[str, Any]:
    if sample.get("split") != "validation":
        raise GenerativeError("Synthetic generation must be conditioned on a real held-out patch.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise GenerativeError("seed must be an integer from 0 through 4294967295.")
    return {
        "label": "synthetic_gan_output_not_cfd_simulation",
        "conditioning_sample_id": sample["sample_id"],
        "source_case_id": sample["case_id"],
        "source_mesh_resolution": sample["mesh_resolution"],
        "source_solution_time": sample["solution_time"],
        "source_split": "validation",
        "patch_index": sample["patch_index"],
        "patch_center": sample["patch_center"],
        "patch_center_units": sample["patch_center_units"],
        "generation_seed": seed,
        "field": {"region": "coolant", "name": "T", "units": "K"},
        "status": "synthetic_exploratory",
    }


def _base_metadata(
    model_type: str,
    model_id: str,
    dataset: dict[str, Any],
    seed: int,
    epochs: int,
    mean: float,
    scale: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "model_type": model_type,
        "status": "trained_exploratory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "case_id": dataset["samples"][0]["case_id"],
            "reynolds_number": dataset["samples"][0]["reynolds_number"],
            "mesh_resolution": dataset["samples"][0]["mesh_resolution"],
            "solution_time": dataset["samples"][0]["solution_time"],
            "target": dataset["target"],
            "cell_count": dataset["cell_count"],
            "manifest_sha256": dataset["manifest_sha256"],
        },
        "representation": dataset["representation"],
        "split": {
            "method": "spatial x-slab holdout with non-overlapping buffered region",
            "patch_counts": dataset["counts"],
            "training_patch_origins_x_max": PATCH_SIZE // 2,
            "validation_patch_origins_x_min": GRID_SIZE - PATCH_SIZE,
            "buffer_patch_origins": "all remaining x origins",
        },
        "configuration": {
            "seed": seed,
            "epochs": epochs,
            "device": "cpu",
            "latent_dimension": LATENT_DIMENSION,
            "hidden_dimension": HIDDEN_DIMENSION,
            "patch_shape": [PATCH_SIZE, PATCH_SIZE, PATCH_SIZE],
            "normalization": {
                "method": "training-valid-voxel mean and standard deviation",
                "mean_K": mean,
                "standard_deviation_K": scale,
            },
            "vae_beta": VAE_BETA if model_type == "vae" else None,
        },
        "limitations": [
            "This is a small single-case pilot, not a validated CFD surrogate or new simulation.",
            "The regular voxel representation is a coarse, piecewise-constant projection of an unstructured mesh.",
            "Only spatially separated patches from one case are evaluated; generalization across cases is unknown.",
            "Synthetic similarity and temperature positivity do not establish physical validity or safe cooling designs.",
        ],
    }


def train_generative_model(
    dataset: dict[str, Any],
    *,
    model_type: str,
    seed: int,
    epochs: int,
    models_root: Path,
) -> dict[str, Any]:
    model_type, seed, epochs = validate_training_config(model_type, seed, epochs)
    torch = require_torch()
    if (
        dataset.get("target")
        != {
            "region": "coolant",
            "field": "T",
            "units": "K",
            "field_class": "volScalarField",
            "association": "cell",
        }
        or not isinstance(dataset.get("manifest_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", dataset["manifest_sha256"])
        or not isinstance(dataset.get("cell_count"), int)
        or not isinstance(dataset.get("samples"), list)
        or not dataset["samples"]
    ):
        raise GenerativeError("Training requires provenance-verified coolant/T patches from a scanned dataset.")
    values, masks, split_labels = _patch_arrays(dataset)
    train_indexes, validation_indexes = _selected_indexes(split_labels)
    training_values = values[train_indexes]
    training_masks = masks[train_indexes]
    validation_values = values[validation_indexes]
    validation_masks = masks[validation_indexes]
    mean, scale = _normalization(training_values, training_masks)
    train_x, train_masks_tensor = _to_tensor(torch, training_values, training_masks, mean, scale)
    validation_x, validation_masks_tensor = _to_tensor(
        torch, validation_values, validation_masks, mean, scale
    )
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    VAE, Generator, Discriminator = _new_networks(torch)
    if model_type == "vae":
        model = VAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output, latent_mean, latent_log_variance = model(
                train_x, train_masks_tensor, sample=True
            )
            mask_total = train_masks_tensor.sum().clamp_min(1)
            reconstruction_loss = (
                ((output - train_x) ** 2 * train_masks_tensor).sum() / mask_total
            )
            kl_loss = -0.5 * torch.mean(
                1 + latent_log_variance - latent_mean.pow(2) - latent_log_variance.exp()
            )
            loss = reconstruction_loss + VAE_BETA * kl_loss
            loss.backward()
            optimizer.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": float(loss.detach()),
                    "training_masked_mse_normalized": float(reconstruction_loss.detach()),
                    "training_kl": float(kl_loss.detach()),
                }
            )
        model.eval()
        with torch.no_grad():
            validation_output, _, _ = model(
                validation_x, validation_masks_tensor, sample=False
            )
        reconstructed = (
            validation_output.numpy().reshape(validation_values.shape) * scale + mean
        )
        metrics = reconstruction_metrics(validation_values, reconstructed, validation_masks)
        metrics.update(positive_temperature_check(reconstructed, validation_masks))
        network_states = {"vae": model.state_dict()}
    else:
        generator = Generator()
        discriminator = Discriminator()
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        loss_function = torch.nn.BCEWithLogitsLoss()
        history = []
        for epoch in range(epochs):
            generator.train()
            discriminator.train()
            noise = torch.randn((train_x.shape[0], LATENT_DIMENSION), generator=rng)
            fake_values = generator(noise, train_masks_tensor)
            real_logits = discriminator(train_x, train_masks_tensor)
            fake_logits = discriminator(fake_values.detach(), train_masks_tensor)
            discriminator_loss = loss_function(real_logits, torch.ones_like(real_logits)) + loss_function(
                fake_logits, torch.zeros_like(fake_logits)
            )
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            noise = torch.randn((train_x.shape[0], LATENT_DIMENSION), generator=rng)
            fake_values = generator(noise, train_masks_tensor)
            generated_logits = discriminator(fake_values, train_masks_tensor)
            generator_loss = loss_function(generated_logits, torch.ones_like(generated_logits))
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "generator_loss": float(generator_loss.detach()),
                    "discriminator_loss": float(discriminator_loss.detach()),
                }
            )
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            noise = torch.randn(
                (validation_masks_tensor.shape[0], LATENT_DIMENSION), generator=rng
            )
            validation_generated = (
                generator(noise, validation_masks_tensor).numpy().reshape(validation_values.shape)
                * scale
                + mean
            )
        metrics = evaluate_generated_samples(
            validation_generated,
            validation_masks,
            validation_values,
            validation_masks,
        )
        network_states = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }

    manifest_sha256 = dataset["manifest_sha256"]
    model_id = _model_id(
        model_type,
        dataset["samples"][0]["case_id"],
        manifest_sha256,
        seed,
        epochs,
    )
    metadata = _base_metadata(model_type, model_id, dataset, seed, epochs, mean, scale)
    metadata.update(
        {
            "metrics": metrics,
            "training_history": history,
            "training_history_summary": {
                "record_count": len(history),
                "final": history[-1],
            },
        }
    )
    model_directory = models_root / "generative" / "models" / model_id
    checkpoint_path = model_directory / "checkpoint.pt"
    checkpoint = {
        "schema_version": 1,
        "model_type": model_type,
        "configuration": metadata["configuration"],
        "normalization": metadata["configuration"]["normalization"],
        "network_states": network_states,
    }
    _save_checkpoint(torch, checkpoint_path, checkpoint)
    try:
        _atomic_json(model_directory / "metadata.json", metadata)
    except OSError:
        checkpoint_path.unlink(missing_ok=True)
        raise
    return metadata


def load_model_metadata(models_root: Path, model_id: str) -> dict[str, Any]:
    metadata_path, checkpoint_path = _model_paths(models_root, model_id)
    if not metadata_path.is_file():
        raise GenerativeError(
            f"The saved generative model {model_id} is unavailable because metadata.json is missing; "
            "restore the model directory or train a new model.",
            404,
        )
    if not checkpoint_path.is_file():
        raise GenerativeError(
            f"The saved generative model {model_id} is unavailable because checkpoint.pt is missing; "
            "restore the checkpoint or train a new model.",
            404,
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GenerativeError("The saved generative model metadata is malformed.", 500) from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_id") != model_id
        or metadata.get("schema_version") != 1
        or metadata.get("model_type") not in {"vae", "gan"}
        or not isinstance(metadata.get("source"), dict)
    ):
        raise GenerativeError("The saved generative model metadata does not match its checkpoint.", 409)
    return metadata


def _load_model(models_root: Path, model_id: str, manifest_sha256: str):
    metadata = load_model_metadata(models_root, model_id)
    if metadata.get("source", {}).get("manifest_sha256") != manifest_sha256:
        raise GenerativeError("The dataset manifest changed after training; retrain this model.", 409)
    torch = require_torch()
    _metadata_path, checkpoint_path = _model_paths(models_root, model_id)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise GenerativeError("The saved generative checkpoint could not be loaded.", 500) from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("model_type") != metadata.get("model_type")
    ):
        raise GenerativeError("The saved generative checkpoint is incompatible with its metadata.", 409)
    VAE, Generator, Discriminator = _new_networks(torch)
    if metadata["model_type"] == "vae":
        model = VAE()
        try:
            model.load_state_dict(checkpoint["network_states"]["vae"])
        except (KeyError, RuntimeError) as error:
            raise GenerativeError("The VAE checkpoint is missing compatible model weights.", 409) from error
        model.eval()
    else:
        model = Generator()
        try:
            model.load_state_dict(checkpoint["network_states"]["generator"])
        except (KeyError, RuntimeError) as error:
            raise GenerativeError("The GAN checkpoint is missing compatible generator weights.", 409) from error
        model.eval()
    return torch, model, metadata


def list_generative_models(models_root: Path, current_manifest_sha256: str) -> list[dict[str, Any]]:
    root = models_root / "generative" / "models"
    if not root.exists():
        return []
    models = []
    for metadata_path in root.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                continue
            model_id = metadata.get("model_id")
            if not isinstance(model_id, str) or not MODEL_ID_PATTERN.fullmatch(model_id):
                continue
            _metadata, checkpoint = _model_paths(models_root, model_id)
            if not checkpoint.is_file():
                metadata["available"] = False
                metadata["unavailable_reason"] = (
                    f"checkpoint.pt is missing for {model_id}; restore the checkpoint or train a new model."
                )
                models.append(metadata)
                continue
            metadata["available"] = (
                metadata.get("source", {}).get("manifest_sha256") == current_manifest_sha256
            )
            if not metadata["available"]:
                metadata["unavailable_reason"] = "The dataset manifest changed after training."
            models.append(metadata)
        except (OSError, json.JSONDecodeError, GenerativeError):
            continue
    return sorted(models, key=lambda item: item.get("created_at", ""), reverse=True)


def _safe_array(values: np.ndarray, masks: np.ndarray) -> list[Any]:
    return [
        [
            [float(values[x, y, z]) if masks[x, y, z] else None for z in range(values.shape[2])]
            for y in range(values.shape[1])
        ]
        for x in range(values.shape[0])
    ]


def reconstruct_validation_sample(
    dataset: dict[str, Any], *, models_root: Path, model_id: str, sample_id: str, manifest_sha256: str
) -> dict[str, Any]:
    torch, model, metadata = _load_model(models_root, model_id, manifest_sha256)
    if metadata["model_type"] != "vae":
        raise GenerativeError("Select a VAE model to reconstruct a real held-out patch.")
    sample_index = next(
        (
            index
            for index, sample in enumerate(dataset["samples"])
            if sample["sample_id"] == sample_id and sample["split"] == "validation"
        ),
        None,
    )
    if sample_index is None:
        raise GenerativeError("The selected real sample is not in the model's held-out spatial region.", 404)
    values = np.asarray(dataset["values"][sample_index], dtype=np.float32)
    masks = np.asarray(dataset["masks"][sample_index], dtype=bool)
    normalization = metadata["configuration"]["normalization"]
    x, mask_tensor = _to_tensor(torch, values[None], masks[None], normalization["mean_K"], normalization["standard_deviation_K"])
    with torch.no_grad():
        normalized, _, _ = model(x, mask_tensor, sample=False)
    reconstructed = (
        normalized.numpy().reshape(values.shape) * normalization["standard_deviation_K"]
        + normalization["mean_K"]
    )
    reconstructed[~masks] = 0
    actual = np.asarray(values, dtype=np.float64)
    difference = reconstructed.astype(np.float64) - actual
    metrics = reconstruction_metrics(actual, reconstructed, masks)
    metrics.update(positive_temperature_check(reconstructed, masks))
    return {
        "label": "VAE reconstruction of a held-out real CFD patch",
        "sample": dataset["samples"][sample_index],
        "target": metadata["source"]["target"],
        "values_K": {
            "original": _safe_array(actual, masks),
            "reconstructed": _safe_array(reconstructed, masks),
            "difference": _safe_array(difference, masks),
        },
        "mask": masks.tolist(),
        "metrics": metrics,
        "status": "exploratory",
    }


def generate_synthetic_sample(
    dataset: dict[str, Any],
    *,
    models_root: Path,
    model_id: str,
    sample_id: str,
    manifest_sha256: str,
    seed: int,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise GenerativeError("seed must be an integer from 0 through 4294967295.")
    torch, generator, metadata = _load_model(models_root, model_id, manifest_sha256)
    if metadata["model_type"] != "gan":
        raise GenerativeError("Select a GAN model to generate an explicitly synthetic field patch.")
    sample_index = next(
        (
            index
            for index, sample in enumerate(dataset["samples"])
            if sample["sample_id"] == sample_id and sample["split"] == "validation"
        ),
        None,
    )
    validation_indexes = np.flatnonzero(np.asarray(dataset["split_labels"]) == "validation")
    if sample_index is None or validation_indexes.size == 0:
        raise GenerativeError("The selected real mask or spatial holdout sample is unavailable.", 404)
    masks = np.asarray(dataset["masks"], dtype=bool)
    values = np.asarray(dataset["values"], dtype=np.float64)
    selected_mask = masks[sample_index]
    normalization = metadata["configuration"]["normalization"]
    masks_tensor = torch.as_tensor(selected_mask.reshape(1, 512).astype(np.float32))
    random = torch.Generator(device="cpu").manual_seed(seed)
    generator_input = torch.randn((1, LATENT_DIMENSION), generator=random)
    with torch.no_grad():
        normalized = generator(generator_input, masks_tensor).numpy().reshape(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    generated = normalized * normalization["standard_deviation_K"] + normalization["mean_K"]
    generated[~selected_mask] = 0

    evaluation_masks = masks[validation_indexes]
    evaluation_real = values[validation_indexes]
    evaluation_random = torch.Generator(device="cpu").manual_seed(seed + 1)
    evaluation_noise = torch.randn(
        (len(validation_indexes), LATENT_DIMENSION), generator=evaluation_random
    )
    evaluation_masks_tensor = torch.as_tensor(
        evaluation_masks.reshape(-1, 512).astype(np.float32)
    )
    with torch.no_grad():
        evaluation_generated = (
            generator(evaluation_noise, evaluation_masks_tensor)
            .numpy()
            .reshape(evaluation_masks.shape)
            * normalization["standard_deviation_K"]
            + normalization["mean_K"]
        )
    evaluation_generated[~evaluation_masks] = 0
    metrics = evaluate_generated_samples(
        evaluation_generated, evaluation_masks, evaluation_real, evaluation_masks
    )
    synthetic_metadata = generated_sample_metadata(
        dataset["samples"][sample_index], seed
    )
    return {
        "label": "Synthetic GAN-generated field patch (not a CFD simulation)",
        "synthetic_sample": synthetic_metadata,
        "sample": dataset["samples"][sample_index],
        "target": metadata["source"]["target"],
        "values_K": _safe_array(generated, selected_mask),
        "mask": selected_mask.tolist(),
        "generation_seed": seed,
        "metrics": metrics,
        "status": "synthetic_exploratory",
        "limitations": metadata["limitations"],
    }
