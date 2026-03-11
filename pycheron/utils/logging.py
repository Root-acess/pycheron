"""
pycheron.utils.logging — Structured logging with Rich formatting.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _RICH = True
except ImportError:
    _RICH = False

_console = Console(stderr=True) if _RICH else None
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger for the given module name.

    Parameters
    ----------
    name : str
        Usually __name__ of the calling module.
    level : str, optional
        Override log level (DEBUG, INFO, WARNING, ERROR).
    """
    if name in _loggers:
        return _loggers[name]

    from pycheron.config.settings import config

    log_level = level or config.get("log_level", "INFO")
    numeric = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric)

    if not logger.handlers:
        if _RICH:
            handler = RichHandler(
                console=_console,
                show_time=False,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
        else:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("%(levelname)s | %(name)s | %(message)s")
            )
        handler.setLevel(numeric)
        logger.addHandler(handler)
        logger.propagate = False

    _loggers[name] = logger
    return logger


class StageLogger:
    """Context manager that logs start/end of a pipeline stage."""

    def __init__(self, stage: str, verbose: int = 1) -> None:
        self.stage = stage
        self.verbose = verbose
        self.logger = get_logger("pycheron.pipeline")

    def __enter__(self) -> "StageLogger":
        if self.verbose >= 1:
            self.logger.info(f"[bold cyan]▶ {self.stage}[/bold cyan]")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            if self.verbose >= 1:
                self.logger.info(f"[bold green]✓ {self.stage} complete[/bold green]")
        else:
            self.logger.error(f"[bold red]✗ {self.stage} failed: {exc_val}[/bold red]")
        return False
