from __future__ import annotations

from pathlib import Path

from jit_sdp_analysis import (
    Stage1UnivariateAnalyzer,
    Stage2MDIAnalyzer,
    Stage3PerformanceAnalyzer,
    Stage4CategoryComparisonAnalyzer,
    all_commit_level_dataset_specs,
    commit_level_feature_config,
    DatasetRepository,
)


OUTPUT_ROOT = Path("experiment_analysis_outputs")


def main() -> None:
    print("[main] Loading datasets...", flush=True)
    datasets = DatasetRepository(
        all_commit_level_dataset_specs(),
        commit_level_feature_config(),
    ).load_all()
    print(f"[main] Loaded {len(datasets)} datasets.", flush=True)
    for dataset in datasets:
        print(
            f"[main] {dataset.name}: {len(dataset.frame)} rows, "
            f"{len(dataset.feature_columns)} features.",
            flush=True,
        )

    print("[stage1] Starting univariate analysis...", flush=True)
    stage1 = Stage1UnivariateAnalyzer()
    stage1_result = stage1.run(
        datasets,
        output_dir=OUTPUT_ROOT / "stage1",
    )
    print(f"[stage1] Finished. Ranked concepts: {len(stage1_result.consensus_ranking)}", flush=True)

    print("[stage2] Starting full-feature MDI analysis...", flush=True)
    stage2 = Stage2MDIAnalyzer()
    stage2_result = stage2.run(
        datasets,
        output_dir=OUTPUT_ROOT / "stage2",
    )
    print("[stage2] Finished.", flush=True)

    print("[stage3] Starting fixed top-k feature subset performance experiment...", flush=True)
    stage3 = Stage3PerformanceAnalyzer()
    stage3.run_topk_experiment(
        datasets,
        stage2_result.stage2b_mdi_npsk_rank_table,
        output_dir=OUTPUT_ROOT / "stage3" / "topk",
    )
    print("[stage3] Finished.", flush=True)

    print("[stage4] Starting Kamei plus feature-category comparison experiment...", flush=True)
    stage4 = Stage4CategoryComparisonAnalyzer()
    stage4.run(
        datasets,
        output_dir=OUTPUT_ROOT / "stage4" / "feature_category_comparison",
    )
    print("[stage4] Finished.", flush=True)

    print(f"[done] Outputs written under: {OUTPUT_ROOT.resolve()}", flush=True)


if __name__ == "__main__":
    main()
