"""Repository path resolution and commit snapshot helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SnapshotContext:
    """Temporary commit snapshot produced via git worktree."""

    repo_path: Path
    commit_hash: str
    snapshot_path: Path
    temp_root: tempfile.TemporaryDirectory

    def cleanup(self) -> None:
        """Remove the temporary worktree and temp directory."""
        try:
            subprocess.run(
                ["git", "-C", str(self.repo_path), "worktree", "remove", "--force", str(self.snapshot_path)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        finally:
            self.temp_root.cleanup()


def create_snapshot_at_commit(repo_path: Path, commit_hash: str) -> SnapshotContext:
    """Create an explicit repository snapshot for the given commit using git worktree."""
    temp_root = tempfile.TemporaryDirectory(prefix="commit_snapshot_")
    snapshot_path = Path(temp_root.name) / "snapshot"

    subprocess.run(
        ["git", "-C", str(repo_path), "worktree", "add", "--detach", str(snapshot_path), commit_hash],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return SnapshotContext(
        repo_path=repo_path,
        commit_hash=commit_hash,
        snapshot_path=snapshot_path,
        temp_root=temp_root,
    )
def clone_github_repo_to_local(repo_url: str, local_repo_path: Path, logger) -> Path:
    """Clone a GitHub repository via git, then copy it to the target local path."""
    local_repo_path = local_repo_path.expanduser().resolve()
    local_repo_path.parent.mkdir(parents=True, exist_ok=True)

    if local_repo_path.exists():
        raise FileExistsError(
            f"Target local repository path already exists: {local_repo_path}. "
            "Please provide a non-existing directory for --local_repo_path."
        )

    with tempfile.TemporaryDirectory(prefix="github_clone_") as temp_dir:
        temp_clone_path = Path(temp_dir) / "repo"
        logger.info("Cloning GitHub repository: %s", repo_url)
        subprocess.run(
            ["git", "clone", repo_url, str(temp_clone_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        logger.info("Copying repository to local path: %s", local_repo_path)
        shutil.copytree(temp_clone_path, local_repo_path)

    return local_repo_path


def resolve_analysis_repo_path(
    repo_path: Path | None,
    repo_url: str | None,
    local_repo_path: Path | None,
    logger,
) -> Path:
    """Resolve the repository path to analyze."""
    if repo_url:
        if local_repo_path is None:
            raise ValueError("--local_repo_path is required when --repo_url is provided.")
        return clone_github_repo_to_local(repo_url, local_repo_path, logger)

    if repo_path is None:
        raise ValueError("You must provide either --repo_path or --repo_url.")

    resolved_repo_path = repo_path.expanduser().resolve()
    if not resolved_repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {resolved_repo_path}")
    return resolved_repo_path
