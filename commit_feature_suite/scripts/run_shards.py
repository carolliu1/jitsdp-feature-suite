#!/usr/bin/env python3
"""Run commit_feature_suite in multi-process shards and merge outputs."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard runner for commit_feature_suite")
    parser.add_argument("--repo_path", required=True, help="Local repository path")
    parser.add_argument("--output_dir", required=True, help="Directory for shard outputs")
    parser.add_argument("--output_prefix", required=True, help="Output prefix, e.g. spring_boot_feature_suite")
    parser.add_argument("--total_commits", type=int, required=True, help="Total commits to cover")
    parser.add_argument("--shards", type=int, default=8, help="Number of shards/processes")
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=None,
        help="Maximum number of shard subprocesses running concurrently. "
        "Default: same as --shards (legacy behavior).",
    )
    parser.add_argument("--languages", default="auto", help="Language list, e.g. java,kotlin,groovy")
    parser.add_argument("--snapshot_mode", default="current", choices=["current"])
    parser.add_argument("--enable_rca_metrics", default="false", choices=["true", "false"])
    parser.add_argument("--rca_command", default="rust-code-analysis-cli")
    parser.add_argument(
        "--isolate_repo_per_shard",
        default="true",
        choices=["true", "false"],
        help="Use one shared clone per shard to avoid .git/config lock contention. Default: true",
    )
    parser.add_argument(
        "--keep_shard_repos",
        default="false",
        choices=["true", "false"],
        help="Keep shard-local repositories for debugging. Default: false",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def merge_csv_files(files: list[Path], merged_output: Path) -> None:
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with merged_output.open("w", encoding="utf-8", newline="") as out_f:
        writer = None
        for file_path in files:
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.reader(in_f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                if not wrote_header:
                    writer = csv.writer(out_f)
                    writer.writerow(header)
                    wrote_header = True
                if writer is None:
                    writer = csv.writer(out_f)
                for row in reader:
                    writer.writerow(row)


def drop_parent_and_diff_columns(commit_level_csv: Path) -> None:
    """Drop parent_* and diff_* columns from merged commit-level CSV."""
    if not commit_level_csv.exists():
        return
    df = pd.read_csv(commit_level_csv)
    if df.empty:
        df.to_csv(commit_level_csv, index=False, na_rep="None")
        return
    drop_cols = [
        col
        for col in df.columns
        if col.startswith("parent_")
        or col.startswith("diff_")
        or col.endswith("_parent")
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df.to_csv(commit_level_csv, index=False, na_rep="None")
    print(f"Dropped {len(drop_cols)} parent/diff columns from: {commit_level_csv}")


def main() -> int:
    args = parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be a positive integer.")
    if args.parallel_workers is not None and args.parallel_workers <= 0:
        raise ValueError("--parallel_workers must be a positive integer.")

    parallel_workers = args.parallel_workers if args.parallel_workers is not None else args.shards
    parallel_workers = min(parallel_workers, args.shards)

    output_dir = Path(args.output_dir).expanduser().resolve()
    repo_path = Path(args.repo_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    isolate_repo_per_shard = args.isolate_repo_per_shard.lower() == "true"
    keep_shard_repos = args.keep_shard_repos.lower() == "true"
    shard_repo_root = Path(
        tempfile.mkdtemp(prefix="cmc_shards_", dir=str(output_dir))
    ) if isolate_repo_per_shard else None

    per_shard = (args.total_commits + args.shards - 1) // args.shards
    function_files: list[Path] = []
    file_files: list[Path] = []
    commit_files: list[Path] = []
    shard_specs: list[dict] = []

    for shard_index in range(args.shards):
        skip = shard_index * per_shard
        if skip >= args.total_commits:
            break
        limit = per_shard
        out_csv = output_dir / f"{args.output_prefix}_part_{shard_index}.csv"
        function_files.append(output_dir / f"{args.output_prefix}_part_{shard_index}_function_level.csv")
        file_files.append(output_dir / f"{args.output_prefix}_part_{shard_index}_file_level.csv")
        commit_files.append(output_dir / f"{args.output_prefix}_part_{shard_index}_commit_level.csv")
        shard_specs.append(
            {
                "shard_index": shard_index,
                "skip": skip,
                "limit": limit,
                "out_csv": out_csv,
            }
        )

    print(f"Dispatch config: total_shards={len(shard_specs)}, parallel_workers={parallel_workers}")

    running: dict[int, subprocess.Popen] = {}
    next_idx = 0
    completed = 0
    failed_shards: list[tuple[int, int]] = []

    def start_shard(spec: dict) -> subprocess.Popen:
        shard_index = spec["shard_index"]
        skip = spec["skip"]
        limit = spec["limit"]
        out_csv = spec["out_csv"]
        shard_repo_path = repo_path
        if isolate_repo_per_shard:
            assert shard_repo_root is not None
            shard_repo_path = shard_repo_root / f"repo_part_{shard_index}"
            shared_cmd = [
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                str(repo_path),
                str(shard_repo_path),
            ]
            try:
                subprocess.run(
                    shared_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except subprocess.CalledProcessError:
                # Fallback for repositories where shared clone/alternates can
                # trigger missing-object issues in later history traversal.
                fallback_cmd = [
                    "git",
                    "clone",
                    "--no-checkout",
                    str(repo_path),
                    str(shard_repo_path),
                ]
                subprocess.run(
                    fallback_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

        cmd = [
            sys.executable,
            "-m",
            "commit_feature_suite",
            "--repo_path",
            str(shard_repo_path),
            "--output_csv",
            str(out_csv),
            "--skip_commits",
            str(skip),
            "--max_commits",
            str(limit),
            "--languages",
            args.languages,
            "--snapshot_mode",
            args.snapshot_mode,
            "--enable_rca_metrics",
            args.enable_rca_metrics,
            "--rca_command",
            args.rca_command,
            "--log_level",
            args.log_level,
        ]
        print(f"Start shard={shard_index}, skip={skip}, max={limit}, out={out_csv}")
        return subprocess.Popen(cmd)

    while completed < len(shard_specs):
        while len(running) < parallel_workers and next_idx < len(shard_specs):
            spec = shard_specs[next_idx]
            proc = start_shard(spec)
            running[spec["shard_index"]] = proc
            next_idx += 1

        finished: list[int] = []
        for shard_id, proc in running.items():
            code = proc.poll()
            if code is None:
                continue
            print(f"Finish shard={shard_id}, returncode={code}")
            completed += 1
            if code != 0:
                failed_shards.append((shard_id, code))
            finished.append(shard_id)

        for shard_id in finished:
            del running[shard_id]

        if completed < len(shard_specs) and not finished:
            time.sleep(0.5)

    if failed_shards:
        print("One or more shard processes failed.", file=sys.stderr)
        print(f"Failed shards: {failed_shards}", file=sys.stderr)
        if shard_repo_root is not None and not keep_shard_repos:
            shutil.rmtree(shard_repo_root, ignore_errors=True)
        return 1

    merged_function = output_dir / f"{args.output_prefix}_merged_function_level.csv"
    merged_file = output_dir / f"{args.output_prefix}_merged_file_level.csv"
    merged_commit = output_dir / f"{args.output_prefix}_merged_commit_level.csv"
    merge_csv_files(function_files, merged_function)
    merge_csv_files(file_files, merged_file)
    merge_csv_files(commit_files, merged_commit)
    drop_parent_and_diff_columns(merged_commit)
    if shard_repo_root is not None and not keep_shard_repos:
        shutil.rmtree(shard_repo_root, ignore_errors=True)
    print(f"Merged function-level CSV: {merged_function}")
    print(f"Merged file-level CSV: {merged_file}")
    print(f"Merged commit-level CSV: {merged_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

