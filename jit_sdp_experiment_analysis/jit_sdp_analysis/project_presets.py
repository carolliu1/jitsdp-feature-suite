from __future__ import annotations

from pathlib import Path

from .config import DatasetSpec, FeatureFamily, FeatureSelectionConfig


DEFAULT_MERGED_RESULT_DIR = Path("/home/workstation/yihan/datasets")


COMMIT_LEVEL_PROJECT_NAMES: tuple[str, ...] = (
    "camel",
    "django",
    "elasticsearch",
    "godot",
    "mysql-server",
    "node",
    "nova",
    "pandas",
    "pytorch",
    "rust",
    "spring-boot",
    "tensorflow",
    "tomcat",
    "vscode",
    "wp-calypso",
)


KAMEI_FEATURE_COLUMNS: tuple[str, ...] = (
    "fix",
    "ns",
    "nd",
    "nf",
    "entropy",
    "la",
    "ld",
    "lt",
    "ndev",
    "age",
    "nuc",
    "exp",
    "rexp",
    "sexp",
)


GROUPED_COMMIT_LEVEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "current_method_in_coupling_mean",
    "current_method_in_coupling_max",
    "current_method_in_coupling_min",
    "current_method_in_coupling_std",
    "current_method_out_coupling_mean",
    "current_method_out_coupling_max",
    "current_method_out_coupling_min",
    "current_method_out_coupling_std",
    "current_method_cc_mean",
    "current_method_cc_max",
    "current_method_cc_min",
    "current_method_cc_std",
    "current_method_halstead_n1_mean",
    "current_method_halstead_n1_max",
    "current_method_halstead_n1_min",
    "current_method_halstead_n1_std",
    "current_method_halstead_n2_mean",
    "current_method_halstead_n2_max",
    "current_method_halstead_n2_min",
    "current_method_halstead_n2_std",
    "current_method_halstead_N1_mean",
    "current_method_halstead_N1_max",
    "current_method_halstead_N1_min",
    "current_method_halstead_N1_std",
    "current_method_halstead_N2_mean",
    "current_method_halstead_N2_max",
    "current_method_halstead_N2_min",
    "current_method_halstead_N2_std",
    "current_method_halstead_length_mean",
    "current_method_halstead_length_max",
    "current_method_halstead_length_min",
    "current_method_halstead_length_std",
    "current_method_halstead_vocabulary_mean",
    "current_method_halstead_vocabulary_max",
    "current_method_halstead_vocabulary_min",
    "current_method_halstead_vocabulary_std",
    "current_method_halstead_volume_mean",
    "current_method_halstead_volume_max",
    "current_method_halstead_volume_min",
    "current_method_halstead_volume_std",
    "current_method_halstead_difficulty_mean",
    "current_method_halstead_difficulty_max",
    "current_method_halstead_difficulty_min",
    "current_method_halstead_difficulty_std",
    "current_method_halstead_effort_mean",
    "current_method_halstead_effort_max",
    "current_method_halstead_effort_min",
    "current_method_halstead_effort_std",
    "current_method_halstead_bugs_mean",
    "current_method_halstead_bugs_max",
    "current_method_halstead_bugs_min",
    "current_method_halstead_bugs_std",
    "current_method_halstead_time_mean",
    "current_method_halstead_time_max",
    "current_method_halstead_time_min",
    "current_method_halstead_time_std",
    "current_method_nargs_mean",
    "current_method_nargs_max",
    "current_method_nargs_min",
    "current_method_nargs_std",
    "current_method_nexits_mean",
    "current_method_nexits_max",
    "current_method_nexits_min",
    "current_method_nexits_std",
    "current_method_global_var_count_mean",
    "current_method_global_var_count_max",
    "current_method_global_var_count_min",
    "current_method_global_var_count_std",
    "current_file_cloc_mean",
    "current_file_cloc_max",
    "current_file_cloc_min",
    "current_file_cloc_std",
    "current_file_mi_mean",
    "current_file_mi_max",
    "current_file_mi_min",
    "current_file_mi_std",
    "current_file_nom_mean",
    "current_file_nom_max",
    "current_file_nom_min",
    "current_file_nom_std",
)


SINGLE_COMMIT_LEVEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "modified_method_count_current",
    "current_new_file_count",
    "current_new_file_ratio",
    *KAMEI_FEATURE_COLUMNS,
)


COMMIT_LEVEL_FEATURE_COLUMNS: tuple[str, ...] = (
    *GROUPED_COMMIT_LEVEL_FEATURE_COLUMNS,
    *SINGLE_COMMIT_LEVEL_FEATURE_COLUMNS,
)

NEW_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in COMMIT_LEVEL_FEATURE_COLUMNS if column not in KAMEI_FEATURE_COLUMNS
)


def commit_level_feature_config() -> FeatureSelectionConfig:
    return FeatureSelectionConfig(
        include_columns=COMMIT_LEVEL_FEATURE_COLUMNS,
    )


def stage3_feature_families() -> list[FeatureFamily]:
    return [
        FeatureFamily(name="kamei", columns=KAMEI_FEATURE_COLUMNS),
        FeatureFamily(name="new", columns=NEW_FEATURE_COLUMNS),
        FeatureFamily(name="combined", columns=COMMIT_LEVEL_FEATURE_COLUMNS),
    ]


def commit_level_dataset_spec(
    name: str,
    feature_path: str | Path,
    label_column: str = "contains_bug",
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        feature_path=Path(feature_path),
        label_column=label_column,
    )


def camel_dataset_spec() -> DatasetSpec:
    return commit_level_dataset_spec(
        name="camel",
        feature_path=DEFAULT_MERGED_RESULT_DIR / "camel_feature_suite_merged_commit_level.csv",
    )


def all_commit_level_dataset_specs(
    base_dir: str | Path = DEFAULT_MERGED_RESULT_DIR,
    label_column: str = "contains_bug",
) -> list[DatasetSpec]:
    base_path = Path(base_dir)
    return [
        commit_level_dataset_spec(
            name=project_name,
            feature_path=base_path / f"{project_name}_feature_suite_merged_commit_level.csv",
            label_column=label_column,
        )
        for project_name in COMMIT_LEVEL_PROJECT_NAMES
    ]
