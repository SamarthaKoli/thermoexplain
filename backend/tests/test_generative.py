import json

import numpy as np
import pytest

from backend.app import create_app
from backend.config import Settings
from backend.ml import generative
from backend.ml.generative import (
    GenerativeError,
    evaluate_generated_samples,
    generated_sample_metadata,
    load_model_metadata,
    positive_temperature_check,
    reconstruction_metrics,
    train_generative_model,
    validate_training_config,
)
from backend.services import generative_data
from backend.services.generative_data import (
    GenerativeDataError,
    extract_spatial_patches,
    load_prepared_dataset,
    split_patch_origins,
)


def make_prepared_patches():
    grid = np.empty((32, 32, 32), dtype=np.float64)
    x = np.arange(32)[:, None, None]
    y = np.arange(32)[None, :, None]
    z = np.arange(32)[None, None, :]
    grid[:] = 288.0 + x * 0.1 + y * 0.02 - z * 0.01
    mask = np.ones(grid.shape, dtype=bool)
    return extract_spatial_patches(
        grid,
        mask,
        case_id="Re560/coarse",
        reynolds_number=560,
        mesh_resolution="coarse",
        solution_time="200",
        bounds=[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
    )


def test_spatial_patch_preparation_preserves_coordinates_masks_and_buffer():
    prepared = make_prepared_patches()

    assert prepared["values"].shape == (343, 8, 8, 8)
    assert prepared["masks"].shape == prepared["values"].shape
    assert prepared["counts"] == {"train": 98, "validation": 49, "buffer": 196}
    assert prepared["samples"][0]["case_id"] == "Re560/coarse"
    assert prepared["samples"][0]["patch_center_units"] is None
    assert prepared["samples"][0]["patch_index"] == [0, 0, 0]
    assert prepared["samples"][0]["patch_center"] == [0.125, 0.25, 0.375]
    assert "piecewise-constant" in prepared["representation"]["interpolation"]

    train_origins = prepared["origins"][prepared["split_labels"] == "train"]
    heldout_origins = prepared["origins"][prepared["split_labels"] == "validation"]
    assert train_origins[:, 0].max() + 8 < heldout_origins[:, 0].min()


def test_patch_preparation_rejects_malformed_mesh_grid_and_empty_holdout():
    with pytest.raises(GenerativeDataError, match="malformed"):
        extract_spatial_patches(
            np.zeros((8, 8, 8)),
            np.ones((8, 8, 8), dtype=bool),
            case_id="Re560/coarse",
            reynolds_number=560,
            mesh_resolution="coarse",
            solution_time="200",
            bounds=[[0, 1], [0, 1], [0, 1]],
            grid_size=32,
        )
    with pytest.raises(GenerativeDataError, match="too small"):
        extract_spatial_patches(
            np.ones((48, 48, 48)),
            np.ones((48, 48, 48), dtype=bool),
            case_id="Re560/coarse",
            reynolds_number=560,
            mesh_resolution="coarse",
            solution_time="200",
            bounds=[[0, 1], [0, 1], [0, 1]],
            grid_size=48,
            patch_size=16,
            stride=16,
        )


def test_split_rejects_bad_origin_matrix_and_marks_buffer():
    with pytest.raises(GenerativeDataError, match="origins"):
        split_patch_origins(np.zeros((5, 2), dtype=np.int64))
    prepared = make_prepared_patches()
    assert "buffer" in prepared["split_labels"]
    assert set(prepared["split_labels"]) == {"train", "validation", "buffer"}


def test_prepared_patch_cache_round_trips_without_object_arrays(tmp_path):
    prepared = make_prepared_patches()
    prepared.update(
        target={"region": "coolant", "field": "T", "units": "K"},
        cell_count=1000,
        manifest_sha256="verified-test-manifest",
    )
    cache_path = tmp_path / "prepared.npz"
    generative_data._write_cache(cache_path, prepared)

    loaded = load_prepared_dataset(cache_path)

    assert loaded["values"].shape == (343, 8, 8, 8)
    assert loaded["masks"].dtype == np.bool_
    assert loaded["samples"][0]["sample_id"] == "x0_y0_z0"
    assert loaded["manifest_sha256"] == "verified-test-manifest"


@pytest.mark.parametrize(
    ("model_type", "seed", "epochs"),
    [
        ("vae", 42, 30),
        ("gan", 0, 1),
    ],
)
def test_model_training_configuration_accepts_supported_values(model_type, seed, epochs):
    assert validate_training_config(model_type, seed, epochs) == (
        model_type,
        seed,
        epochs,
    )


def test_model_training_configuration_rejects_invalid_type_seed_and_epoch_count():
    for model_type, seed, epochs in [
        ("diffusion", 42, 2),
        ([], 42, 2),
        ("vae", True, 2),
        ("gan", 2**32, 2),
        ("vae", 42, 0),
        ("gan", 42, 101),
    ]:
        with pytest.raises(GenerativeError):
            validate_training_config(model_type, seed, epochs)


def test_training_failure_is_explicit_when_optional_pytorch_is_missing(tmp_path, monkeypatch):
    def unavailable():
        raise GenerativeError("PyTorch test dependency unavailable.", 503)

    monkeypatch.setattr(generative, "require_torch", unavailable)
    with pytest.raises(GenerativeError, match="PyTorch test dependency") as error:
        train_generative_model(
            make_prepared_patches(),
            model_type="vae",
            seed=42,
            epochs=1,
            models_root=tmp_path,
        )
    assert error.value.status_code == 503
    assert not (tmp_path / "generative" / "models").exists()


def test_checkpoint_metadata_loader_rejects_missing_invalid_and_mismatched_models(tmp_path):
    with pytest.raises(GenerativeError, match="invalid"):
        load_model_metadata(tmp_path, "../not-a-model")
    with pytest.raises(GenerativeError, match="unavailable"):
        load_model_metadata(tmp_path, "vae_re560_coarse_0123456789")

    model_directory = (
        tmp_path / "generative" / "models" / "vae_re560_coarse_0123456789"
    )
    model_directory.mkdir(parents=True)
    (model_directory / "checkpoint.pt").write_bytes(b"not loaded by metadata inspection")
    (model_directory / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "model_id": "other"}),
        encoding="utf-8",
    )
    with pytest.raises(GenerativeError, match="does not match"):
        load_model_metadata(tmp_path, model_directory.name)


def test_checkpoint_model_loader_surfaces_runtime_failure(tmp_path, monkeypatch):
    model_id = "vae_re560_coarse_0123456789"
    model_directory = tmp_path / "generative" / "models" / model_id
    model_directory.mkdir(parents=True)
    (model_directory / "checkpoint.pt").write_bytes(b"checkpoint fixture")
    (model_directory / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": model_id,
                "model_type": "vae",
                "source": {"manifest_sha256": "current"},
            }
        ),
        encoding="utf-8",
    )

    def fail_runtime():
        raise GenerativeError("PyTorch checkpoint runtime is unavailable.", 503)

    monkeypatch.setattr(generative, "require_torch", fail_runtime)
    with pytest.raises(GenerativeError, match="runtime is unavailable") as error:
        generative._load_model(tmp_path, model_id, "current")
    assert error.value.status_code == 503


def test_checkpoint_loader_restores_model_weights_without_training(tmp_path):
    if not generative.torch_available():
        pytest.skip("Optional PyTorch is not installed.")
    torch = generative.require_torch()
    model_id = "vae_re560_coarse_abcdef0123"
    model_directory = tmp_path / "generative" / "models" / model_id
    model_directory.mkdir(parents=True)
    vae_type, _generator_type, _discriminator_type = generative._new_networks(torch)
    source_model = vae_type()
    generative._save_checkpoint(
        torch,
        model_directory / "checkpoint.pt",
        {
            "schema_version": 1,
            "model_type": "vae",
            "network_states": {"vae": source_model.state_dict()},
        },
    )
    (model_directory / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": model_id,
                "model_type": "vae",
                "source": {"manifest_sha256": "current"},
            }
        ),
        encoding="utf-8",
    )

    loaded_torch, loaded_model, metadata = generative._load_model(
        tmp_path, model_id, "current"
    )
    values = loaded_torch.zeros((2, 512))
    masks = loaded_torch.ones((2, 512))
    with loaded_torch.no_grad():
        reconstructed, _, _ = loaded_model(values, masks, sample=False)

    assert metadata["model_id"] == model_id
    assert reconstructed.shape == (2, 512)


def test_reconstruction_metrics_use_only_valid_voxels_and_reject_bad_shapes():
    actual = np.array([[[290.0, 291.0]]])
    predicted = np.array([[[289.0, -100.0]]])
    mask = np.array([[[True, False]]])

    assert reconstruction_metrics(actual, predicted, mask) == {
        "mae_K": 1.0,
        "rmse_K": 1.0,
    }
    with pytest.raises(GenerativeError, match="aligned"):
        reconstruction_metrics(actual, predicted[:, :, :1], mask)


def test_kelvin_plausibility_check_uses_only_masked_values():
    values = np.array([[[290.0, -1.0]]])
    masks = np.array([[[True, False]]])

    result = positive_temperature_check(values, masks)

    assert result["positive_kelvin_fraction"] == 1.0
    assert result["positivity_passed"] is True
    failed = positive_temperature_check(-values, masks)
    assert failed["positive_kelvin_fraction"] == 0.0
    assert failed["positivity_passed"] is False


def test_gan_distribution_and_spatial_metrics_and_physical_check_are_measured():
    heldout = np.zeros((1, 2, 2, 2), dtype=np.float64)
    generated = np.ones_like(heldout)
    masks = np.ones_like(heldout, dtype=bool)
    metrics = evaluate_generated_samples(generated, masks, heldout, masks)

    assert metrics["temperature_wasserstein_distance_K"] == 1.0
    assert metrics["generated_mean_neighbor_jump_K_per_voxel"] == 0.0
    assert metrics["heldout_mean_neighbor_jump_K_per_voxel"] == 0.0
    assert metrics["physical_plausibility"]["positivity_passed"] is True

    generated.fill(-1.0)
    failed_check = evaluate_generated_samples(generated, masks, heldout, masks)
    assert failed_check["physical_plausibility"]["positive_kelvin_fraction"] == 0.0
    assert failed_check["physical_plausibility"]["positivity_passed"] is False


def test_generated_sample_metadata_is_explicitly_synthetic_and_tracks_source():
    prepared = make_prepared_patches()
    sample = next(item for item in prepared["samples"] if item["split"] == "validation")
    metadata = generated_sample_metadata(sample, 21)

    assert metadata["label"] == "synthetic_gan_output_not_cfd_simulation"
    assert metadata["source_split"] == "validation"
    assert metadata["source_case_id"] == "Re560/coarse"
    assert metadata["field"] == {"region": "coolant", "name": "T", "units": "K"}
    assert metadata["generation_seed"] == 21
    with pytest.raises(GenerativeError, match="held-out"):
        generated_sample_metadata({**sample, "split": "train"}, 21)


def test_generative_api_reports_dataset_state_and_handles_invalid_training(tmp_path):
    raw = tmp_path / "raw"
    case_path = raw / "Re560" / "coarse"
    case_path.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "scanned",
                "generated_at": "fixture",
                "cases": [
                    {
                        "case_id": "Re560/coarse",
                        "metadata": {
                            "reynolds_number": 560,
                            "mesh_resolution": "coarse",
                            "solution_time": "200",
                        },
                        "fields": [
                            {
                                "region": "coolant",
                                "name": "T",
                                "association": "cell",
                                "field_class": "volScalarField",
                                "units": "K",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw,
        manifest_path=manifest_path,
        audit_report_path=tmp_path / "audit.md",
        processed_path=tmp_path / "processed",
        models_path=tmp_path / "models",
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    status = client.get("/api/generative/status")
    invalid_training = client.post("/api/generative/train", json={"model_type": "vae"})

    assert status.status_code == 200
    assert status.get_json()["cases"][0]["target"]["field"] == "T"
    assert invalid_training.status_code == 400
    assert "case_id" in invalid_training.get_json()["error"]


def test_generative_api_returns_explicit_torch_unavailable_error(tmp_path, monkeypatch):
    from backend.routes import generative as generative_routes

    raw = tmp_path / "raw"
    (raw / "Re560" / "coarse").mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "scanned",
                "cases": [
                    {
                        "case_id": "Re560/coarse",
                        "metadata": {"reynolds_number": 560},
                        "fields": [
                            {
                                "region": "coolant",
                                "name": "T",
                                "association": "cell",
                                "field_class": "volScalarField",
                                "units": "K",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw,
        manifest_path=manifest_path,
        audit_report_path=tmp_path / "audit.md",
        processed_path=tmp_path / "processed",
        models_path=tmp_path / "models",
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    def require_torch():
        raise GenerativeError("PyTorch is not installed.", 503)

    monkeypatch.setattr(generative_routes, "require_torch", require_torch)
    response = client.post(
        "/api/generative/train",
        json={"model_type": "vae", "case_id": "Re560/coarse", "seed": 42, "epochs": 1},
    )

    assert response.status_code == 503
    assert "PyTorch" in response.get_json()["error"]
    assert not (tmp_path / "models" / "generative" / "prepared").exists()
