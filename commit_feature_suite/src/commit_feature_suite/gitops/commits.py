"""Commit traversal helpers based on PyDriller."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pydriller import Repository


def iter_commits(repo_path: Path, max_commits: int | None = None, skip_commits: int = 0) -> Iterator:
    """Yield repository commits in chronological order with optional skip/limit window."""
    commit_iterator = Repository(str(repo_path), only_no_merge=False).traverse_commits()
    yielded = 0
    for index, commit in enumerate(commit_iterator):
        if index < max(0, skip_commits):
            continue
        if max_commits is not None and yielded >= max_commits:
            break
        yielded += 1
        yield commit


def is_merge_commit(commit) -> bool:
    """Return True if the commit is a merge commit."""
    parents = getattr(commit, "parents", []) or []
    return bool(getattr(commit, "merge", False)) or len(parents) > 1
