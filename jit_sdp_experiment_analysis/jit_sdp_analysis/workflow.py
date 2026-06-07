from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    DatasetSpec,
    FeatureSelectionConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
)
from .data import DatasetRepository, PreparedDataset
from .stage1 import Stage1Result, Stage1UnivariateAnalyzer
from .stage2 import Stage2MDIAnalyzer, Stage2Result
from .stage3 import Stage3PerformanceAnalyzer
from .stage4 import Stage4CategoryComparisonAnalyzer


@dataclass
class WorkflowResult:
    datasets: list[PreparedDataset]
    stage1: Stage1Result
    stage2: Stage2Result
    stage3_topk_performance: pd.DataFrame | None = None
    stage4_category_performance: pd.DataFrame | None = None


class FeatureImportanceWorkflow:
    """Orchestrates the full three-stage analysis without becoming a one-off script."""

    def __init__(
        self,
        dataset_specs: list[DatasetSpec] | tuple[DatasetSpec, ...],
        feature_config: FeatureSelectionConfig | None = None,
        stage1_config: Stage1Config | None = None,
        stage2_config: Stage2Config | None = None,
        stage3_config: Stage3Config | None = None,
    ) -> None:
        self.repository = DatasetRepository(dataset_specs, feature_config)
        self.stage1 = Stage1UnivariateAnalyzer(stage1_config)
        self.stage2 = Stage2MDIAnalyzer(stage2_config)
        self.stage3 = Stage3PerformanceAnalyzer(stage3_config)
        self.stage4 = Stage4CategoryComparisonAnalyzer(stage3_config)

    def load(self) -> list[PreparedDataset]:
        return self.repository.load_all()

    def run_all(
        self,
        output_root: Path,
    ) -> WorkflowResult:
        output_root.mkdir(parents=True, exist_ok=True)
        datasets = self.load()

        stage1_result = self.stage1.run(datasets, output_dir=output_root / "stage1")

        stage2_result = self.stage2.run(
            datasets,
            output_dir=output_root / "stage2",
        )

        topk_performance = self.stage3.run_topk_experiment(
            datasets,
            stage2_result.stage2b_mdi_npsk_rank_table,
            output_dir=output_root / "stage3" / "topk",
        )

        category_performance = self.stage4.run(
            datasets,
            output_dir=output_root / "stage4" / "feature_category_comparison",
        )

        return WorkflowResult(
            datasets=datasets,
            stage1=stage1_result,
            stage2=stage2_result,
            stage3_topk_performance=topk_performance,
            stage4_category_performance=category_performance,
        )
