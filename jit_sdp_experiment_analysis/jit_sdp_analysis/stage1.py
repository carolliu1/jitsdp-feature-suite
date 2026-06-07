from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Stage1Config
from .data import PreparedDataset
from .feature_groups import FeatureGrouper
from .metrics import UnivariateScorer
from .npsk import ScottKnottESD
from .plots import (
    family_label,
    plot_stage1_consensus,
    plot_stage1_rank_heatmap,
)


@dataclass
class Stage1Result:
    raw_scores: dict[str, pd.DataFrame]
    concept_scores: dict[str, pd.DataFrame]
    concept_representatives: dict[str, pd.DataFrame]
    npsk_rankings: dict[str, pd.DataFrame]
    consensus_ranking: pd.DataFrame


class Stage1UnivariateAnalyzer:
    """Cross-dataset univariate signal analysis with concept-level max-pooling."""

    def __init__(self, config: Stage1Config | None = None) -> None:
        self.config = config or Stage1Config()

    def run(self, datasets: list[PreparedDataset], output_dir: Path | None = None) -> Stage1Result:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        raw_scores: dict[str, pd.DataFrame] = {}
        concept_scores: dict[str, pd.DataFrame] = {}
        concept_representatives: dict[str, pd.DataFrame] = {}
        npsk_rankings: dict[str, pd.DataFrame] = {}

        for method in self.config.correlation_methods:
            print(f"[stage1] Method {method}: scoring datasets...", flush=True)
            raw_rows = []
            concept_rows = []
            representative_rows = []
            scorer = UnivariateScorer(
                method=method,
                absolute_correlation=self.config.absolute_correlation,
                random_state=self.config.random_state,
            )

            for dataset in datasets:
                grouper = FeatureGrouper(dataset.feature_columns)
                raw = scorer.score(dataset.X, dataset.y)
                concept = grouper.max_pool_scores(raw)
                representatives = grouper.max_pool_representatives(raw)
                representatives.insert(0, "dataset", dataset.name)
                representatives.insert(1, "method", method)
                raw.name = dataset.name
                concept.name = dataset.name
                raw_rows.append(raw)
                concept_rows.append(concept)
                representative_rows.append(representatives)

            raw_matrix = pd.DataFrame(raw_rows)
            concept_matrix = pd.DataFrame(concept_rows)
            representatives_frame = pd.concat(representative_rows, ignore_index=True)
            raw_matrix.index.name = "dataset"
            concept_matrix.index.name = "dataset"
            raw_scores[method] = raw_matrix
            concept_scores[method] = concept_matrix
            concept_representatives[method] = representatives_frame

            print(f"[stage1] Method {method}: running NPSK-ESD...", flush=True)
            ranking = ScottKnottESD(self.config.npsk).rank(concept_matrix, higher_is_better=True)
            ranking["method"] = method
            npsk_rankings[method] = ranking

            if output_dir is not None:
                self._rank_output(ranking).to_csv(output_dir / f"npsk_ranks_{method}.csv", index=False)

        print("[stage1] Building consensus ranking...", flush=True)
        consensus = self._consensus(npsk_rankings)
        if output_dir is not None:
            consensus.to_csv(output_dir / "consensus_ranking.csv", index=False)
            consensus.to_csv(output_dir / "all_concept_npsk_ranks.csv", index=False)
            self._npsk_ranks_long(npsk_rankings).to_csv(
                output_dir / "all_concept_npsk_ranks_long.csv",
                index=False,
            )
            figure_dir = output_dir / "figures"
            plot_stage1_consensus(consensus, figure_dir)
            plot_stage1_rank_heatmap(consensus, figure_dir)

        return Stage1Result(raw_scores, concept_scores, concept_representatives, npsk_rankings, consensus)

    def _rank_output(self, ranking: pd.DataFrame) -> pd.DataFrame:
        columns = ["concept", "rank", "cluster", "n", "method", "feature_family"]
        output = ranking.copy()
        if "feature_family" not in output.columns and "concept" in output.columns:
            output["feature_family"] = output["concept"].map(family_label)
        return output[[column for column in columns if column in output.columns]]

    def _npsk_ranks_long(self, rankings: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for method, ranking in rankings.items():
            frame = ranking.copy()
            frame["method"] = method
            frames.append(self._rank_output(frame))
        if not frames:
            return pd.DataFrame(columns=["method", "concept", "rank", "feature_family"])
        return pd.concat(frames, ignore_index=True)

    def _consensus(self, rankings: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rank_frames = []
        for method, ranking in rankings.items():
            rank_frames.append(ranking[["concept", "rank"]].rename(columns={"rank": method}))
        if not rank_frames:
            return pd.DataFrame(columns=["concept", "average_rank"])

        consensus = rank_frames[0]
        for frame in rank_frames[1:]:
            consensus = consensus.merge(frame, on="concept", how="outer")

        method_columns = list(rankings.keys())
        consensus["average_rank"] = consensus[method_columns].mean(axis=1)
        consensus["rank_std"] = consensus[method_columns].std(axis=1)
        consensus["consensus_rank"] = consensus["average_rank"].rank(method="dense", ascending=True).astype(int)
        consensus["feature_family"] = consensus["concept"].map(family_label)
        return consensus.sort_values(["consensus_rank", "rank_std", "concept"]).reset_index(drop=True)
