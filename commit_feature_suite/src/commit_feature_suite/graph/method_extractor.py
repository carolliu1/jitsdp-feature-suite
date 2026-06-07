"""Method extraction based on lizard."""

from __future__ import annotations

from pathlib import Path
from typing import List

import lizard

from commit_feature_suite.models import MethodInfo


class LizardMethodExtractor:
    """Extract method definitions using lizard."""

    def __init__(self, logger) -> None:
        self.logger = logger

    def extract(self, file_path: Path, relative_path: str, language: str) -> List[MethodInfo]:
        """Extract methods from one source file."""
        methods: List[MethodInfo] = []
        try:
            analysis = lizard.analyze_file(str(file_path))
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to run lizard on %s: %s", relative_path, exc)
            return methods

        for func in getattr(analysis, "function_list", []) or []:
            start_line = int(getattr(func, "start_line", 0) or 0)
            end_line = int(getattr(func, "end_line", 0) or start_line)
            if start_line <= 0:
                continue
            method_name_raw = str(getattr(func, "name", "") or "")
            long_name = str(getattr(func, "long_name", "") or "")
            class_name, method_name = self.split_lizard_method_name(method_name_raw, long_name)
            methods.append(
                MethodInfo(
                    file_path=relative_path,
                    class_name=class_name,
                    method_name=method_name,
                    start_line=start_line,
                    end_line=max(end_line, start_line),
                    language=language,
                )
            )
        return methods

    @staticmethod
    def split_lizard_method_name(name: str, long_name: str) -> tuple[str, str]:
        """Split lizard method signature into class and method name."""
        text = (long_name or name).strip()
        if not text:
            return "", "unknown_method"

        for sep in ("::", ".", "#"):
            if sep in text:
                head, tail = text.rsplit(sep, 1)
                class_name = head.split(sep)[-1].strip()
                method_name = LizardMethodExtractor.normalize_method_name(tail.strip())
                if method_name:
                    return class_name, method_name
        return "", LizardMethodExtractor.normalize_method_name(text)

    @staticmethod
    def normalize_method_name(method_name: str) -> str:
        """Normalize method name from lizard output."""
        name = method_name.strip()
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        if "<" in name and ">" in name and "." not in name:
            name = name.split("<", 1)[0].strip()
        return name or "unknown_method"


