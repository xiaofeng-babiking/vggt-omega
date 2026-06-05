"""Colorful, formatted logging for VGGT-Omega.

Wraps the stdlib :mod:`logging` with a :mod:`colorlog` formatter so console
output reads as::

    [INFO] | [2026-06-03 12:00:00.123] | [logger.py:42] | message goes here

Each field carries its own color: the level name is colored by severity, while
the timestamp and line number get fixed accent colors so they stay legible at a
glance. Call :func:`get_logger` to obtain a configured, idempotent logger.
"""
from __future__ import annotations

import logging
import sys

import colorlog

__all__ = ["get_logger"]

DEFAULT_NAME = "vggt_omega"

# ``%(...)s`` placeholders are wrapped with colorlog color tokens. ``log_color``
# tracks the record severity; the ``time``/``line`` secondary colors are fixed
# (same value for every level) so the timestamp and line number read uniformly.
_LOG_FORMAT = (
    "%(log_color)s[%(levelname)s]%(reset)s | "
    "%(time_log_color)s[%(asctime)s.%(msecs)03d]%(reset)s | "
    "%(line_log_color)s[%(filename)s:%(lineno)d]%(reset)s | "
    "%(message_log_color)s%(message)s%(reset)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Severity palette for the ``[LEVEL]`` field.
_LEVEL_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_white,bg_red",
}

# Fixed accent colors for the remaining fields, keyed the same for every level.
_SECONDARY_COLORS = {
    "time": {level: "blue" for level in _LEVEL_COLORS},
    "line": {level: "purple" for level in _LEVEL_COLORS},
    # Message stays neutral for normal levels but inherits warning/error tones.
    "message": {
        "DEBUG": "white",
        "INFO": "white",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
}


def _build_formatter() -> colorlog.ColoredFormatter:
    """Construct the shared colored formatter."""
    return colorlog.ColoredFormatter(
        _LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        log_colors=_LEVEL_COLORS,
        secondary_log_colors=_SECONDARY_COLORS,
        reset=True,
        style="%",
    )


def get_logger(name: str = DEFAULT_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a colorized logger.

    The logger is configured once per name: repeated calls reuse the existing
    stream handler instead of stacking duplicates, so importing modules can call
    this freely. Propagation to the root logger is disabled to avoid double
    emission when the root has its own handlers.

    Args:
        name: Logger name; defaults to the package name.
        level: Logging level applied to the logger and its handler.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_build_formatter())
        logger.addHandler(handler)

    # Keep the level in sync with the latest request on subsequent calls.
    for handler in logger.handlers:
        handler.setLevel(level)

    return logger


if __name__ == "__main__":
    _demo = get_logger("vggt_omega.demo", level=logging.DEBUG)
    _demo.debug("debug message — verbose diagnostics")
    _demo.info("info message — normal operation")
    _demo.warning("warning message — something looks off")
    _demo.error("error message — an operation failed")
    _demo.critical("critical message — unrecoverable state")
