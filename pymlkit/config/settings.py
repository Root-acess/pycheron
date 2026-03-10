"""
pymlkit.config.settings — Global configuration registry.

Users can override defaults via:
  - pymlkit.config.set('n_jobs', 4)
  - Environment variables: PYMLKIT_N_JOBS=4
  - A pymlkit.toml file in the project root
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

# Try stdlib tomllib (3.11+) then fallback to tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore


_DEFAULTS: Dict[str, Any] = {
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 1,
    "cache_dir": ".pymlkit_cache",
    "log_level": "INFO",
    "test_size": 0.2,
    "cv": 5,
    "tune_trials": 30,
    "time_budget": None,
    "auto_explain": False,
    "auto_save": False,
}

_ENV_MAP = {
    "PYMLKIT_N_JOBS": ("n_jobs", int),
    "PYMLKIT_RANDOM_STATE": ("random_state", int),
    "PYMLKIT_VERBOSE": ("verbose", int),
    "PYMLKIT_CACHE_DIR": ("cache_dir", str),
    "PYMLKIT_LOG_LEVEL": ("log_level", str),
}


class Config:
    """Global configuration object. Access via pymlkit.config."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = dict(_DEFAULTS)
        self._load_env()
        self._load_toml()

    def _load_env(self) -> None:
        for env_key, (cfg_key, cast) in _ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    self._data[cfg_key] = cast(val)
                except (ValueError, TypeError):
                    pass

    def _load_toml(self) -> None:
        if tomllib is None:
            return
        toml_path = Path("pymlkit.toml")
        if not toml_path.exists():
            return
        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            for k, v in data.get("pymlkit", {}).items():
                if k in self._data:
                    self._data[k] = v
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key not in _DEFAULTS:
            raise KeyError(f"Unknown config key: '{key}'. Valid keys: {list(_DEFAULTS)}")
        self._data[key] = value

    def __repr__(self) -> str:
        lines = ["Config("]
        for k, v in self._data.items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# Singleton instance
config = Config()
