"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from commit_feature_suite.config import RuntimeConfig


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> RuntimeConfig:
    """Parse CLI arguments into RuntimeConfig."""
    parser = argparse.ArgumentParser(
        description="Compute commit-level, function-level, and file-level features from Git commit snapshots."
    )
    parser.add_argument("--repo_path", help="Path to an existing local Git repository.")
    parser.add_argument("--repo_url", help="GitHub repository URL to clone with git before analysis.")
    parser.add_argument("--local_repo_path", help="Destination path for repository cloned from --repo_url.")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path.")
    parser.add_argument("--skip_commits", type=int, default=0, help="Skip first N commits before analysis.")
    parser.add_argument("--max_commits", type=int, default=None, help="Analyze at most N commits.")
    parser.add_argument(
        "--languages",
        default="auto",
        help="Comma-separated language list or 'auto'. Default: auto (detect from repository).",
    )
    parser.add_argument(
        "--snapshot_mode",
        default="current",
        choices=["current"],
        help="Snapshot mode: current only.",
    )
    parser.add_argument(
        "--enable_rca_metrics",
        default=False,
        type=_str_to_bool,
        help="Enable rust-code-analysis metrics extraction for affected methods and modified files.",
    )
    parser.add_argument(
        "--rca_command",
        default="rust-code-analysis-cli",
        help="rust-code-analysis executable command path/name. Default: rust-code-analysis-cli",
    )
    parser.add_argument(
        "--rca_debug_dump_dir",
        default=None,
        help="Optional directory to dump raw rust-code-analysis JSON outputs for verification.",
    )
    parser.add_argument("--log_level", default="INFO", help="Logging level. Default: INFO")
    args = parser.parse_args()

    languages = [item.strip().lower() for item in args.languages.split(",") if item.strip()]
    if languages == ["auto"]:
        languages = []
    return RuntimeConfig(
        repo_path=Path(args.repo_path).expanduser().resolve() if args.repo_path else None,
        repo_url=args.repo_url,
        local_repo_path=Path(args.local_repo_path).expanduser().resolve() if args.local_repo_path else None,
        output_csv=Path(args.output_csv).expanduser().resolve(),
        max_commits=args.max_commits,
        skip_commits=max(0, int(args.skip_commits)),
        languages=languages,
        snapshot_mode=args.snapshot_mode,
        enable_rca_metrics=args.enable_rca_metrics,
        rca_command=args.rca_command,
        rca_debug_dump_dir=Path(args.rca_debug_dump_dir).expanduser().resolve() if args.rca_debug_dump_dir else None,
        log_level=args.log_level,
    )


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    from commit_feature_suite.analyzer import CommitFeatureSuiteAnalyzer

    analyzer = CommitFeatureSuiteAnalyzer(config)
    analyzer.run()


if __name__ == "__main__":
    main()

