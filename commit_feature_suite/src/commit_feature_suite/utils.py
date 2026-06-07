"""Shared utility helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from commit_feature_suite.config import LANGUAGE_FILE_EXTENSIONS, SPECIAL_FILE_LANGUAGE_MAP


LOGGER_NAME = "commit_feature_suite"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger


def ensure_parent_dir(file_path: Path) -> None:
    """Create parent directory for a file path if it does not exist."""
    file_path.parent.mkdir(parents=True, exist_ok=True)


def normalize_rel_path(base_path: Path, target_path: Path) -> str:
    """Return POSIX relative path from base_path to target_path."""
    return target_path.relative_to(base_path).as_posix()


def detect_language_from_path(file_path: Path, languages: Iterable[str]) -> str | None:
    """Infer supported language by file extension."""
    supported = set(languages)
    special_language = SPECIAL_FILE_LANGUAGE_MAP.get(file_path.name)
    if special_language and special_language in supported:
        return special_language

    suffix = file_path.suffix.lower()
    for language in languages:
        if suffix in LANGUAGE_FILE_EXTENSIONS.get(language, set()):
            return language
    return None


def is_source_file(file_path: Path, languages: Iterable[str]) -> bool:
    """Return True when file_path matches supported source file extensions."""
    return detect_language_from_path(file_path, languages) is not None


