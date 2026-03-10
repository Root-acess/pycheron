"""
pymlkit.persist — Model saving, loading, and versioning.

A saved model bundle contains:
  model.pkl       — cloudpickle of TrainedModel (estimator + preprocessor)
  manifest.json   — metadata: algorithm, version, scores, feature names, hash
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING, Union

try:
    import cloudpickle as _pickle
except ImportError:
    import pickle as _pickle  # type: ignore

from pymlkit.utils.logging import get_logger

if TYPE_CHECKING:
    from pymlkit.train.trained_model import TrainedModel

logger = get_logger(__name__)

MANIFEST_FILE = "manifest.json"
MODEL_FILE = "model.pkl"


class ModelStore:
    """
    Static class for saving and loading TrainedModel objects.
    """

    @staticmethod
    def save(model: "TrainedModel", path: Union[str, Path]) -> Path:
        """
        Persist a TrainedModel to a directory.

        Creates or increments a version if the directory already exists.

        Parameters
        ----------
        model : TrainedModel
        path  : directory path

        Returns
        -------
        Path to the saved model directory.
        """
        import pymlkit

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Version management
        existing = sorted(save_dir.glob("v*/manifest.json"))
        version = len(existing) + 1
        versioned_dir = save_dir / f"v{version}"
        versioned_dir.mkdir(exist_ok=True)

        # Save model artifact
        model_path = versioned_dir / MODEL_FILE
        with open(model_path, "wb") as f:
            _pickle.dump(model, f)

        # Save manifest
        manifest = {
            "pymlkit_version": pymlkit.__version__,
            "algorithm": model.algorithm,
            "task": model.task.value,
            "target": model.target,
            "version": version,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "feature_names": model.feature_names,
            "evaluation": model.evaluation.to_dict() if model.evaluation else None,
            "meta": model.meta,
        }
        manifest_path = versioned_dir / MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        # Write a "latest" symlink pointer file
        latest_file = save_dir / "latest.txt"
        latest_file.write_text(str(versioned_dir))

        logger.info(f"Model saved to {versioned_dir}")
        return versioned_dir

    @staticmethod
    def load(path: Union[str, Path]) -> "TrainedModel":
        """
        Load a TrainedModel from disk.

        Parameters
        ----------
        path : path to the model directory (loads latest version)
               or a specific versioned directory like 'my_model/v2'.

        Returns
        -------
        TrainedModel ready to predict.
        """
        load_path = Path(path)

        # Resolve to versioned dir
        if (load_path / MODEL_FILE).exists():
            versioned_dir = load_path
        else:
            # Try latest.txt pointer
            latest_file = load_path / "latest.txt"
            if latest_file.exists():
                versioned_dir = Path(latest_file.read_text().strip())
            else:
                # Find highest version
                versions = sorted(load_path.glob("v*/model.pkl"))
                if not versions:
                    raise FileNotFoundError(f"No saved model found at: {load_path}")
                versioned_dir = versions[-1].parent

        model_file = versioned_dir / MODEL_FILE
        if not model_file.exists():
            raise FileNotFoundError(f"model.pkl not found in {versioned_dir}")

        with open(model_file, "rb") as f:
            model = _pickle.load(f)

        # Load manifest for info
        manifest_file = versioned_dir / MANIFEST_FILE
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            logger.info(
                f"Loaded model: {manifest.get('algorithm')} "
                f"(v{manifest.get('version')}, saved {manifest.get('saved_at')})"
            )

        return model

    @staticmethod
    def list_versions(path: Union[str, Path]) -> list:
        """List all saved versions of a model."""
        base = Path(path)
        versions = []
        for manifest_path in sorted(base.glob("v*/manifest.json")):
            with open(manifest_path) as f:
                manifest = json.load(f)
            versions.append({
                "version": manifest.get("version"),
                "algorithm": manifest.get("algorithm"),
                "saved_at": manifest.get("saved_at"),
                "score": manifest.get("evaluation", {}).get("primary_score") if manifest.get("evaluation") else None,
                "path": str(manifest_path.parent),
            })
        return versions
