"""Legacy CSV schema definition (compatibility only).

Notes:
- Main pipeline outputs are written by ``commit_feature_suite.output.writers``.
- This module is kept for backward compatibility with older helper paths
  (e.g. ``output.csv_writer.write_rows_to_csv``).
- When output columns change, update both writer and this schema, or migrate
  callers to ``FeatureOutputWriter`` to avoid schema drift.
"""

CSV_COLUMNS = [
    "commit_id",
    "commit_author_date",
    "snapshot_scope",
    "snapshot_commit_id",
    "is_merge_commit",
    "method_coupling_available",
    "method_id",
    "file_path",
    "class_name",
    "method_name",
    "start_line",
    "end_line",
    "method_in_coupling",
    "method_out_coupling",
    "old_path",
    "new_path",
    "language",
    "node_count",
    "file_count",
    "token_count",
    "method_cc",
    "method_halstead",
    "method_halstead_n1",
    "method_halstead_n2",
    "method_halstead_N1",
    "method_halstead_N2",
    "method_halstead_length",
    "method_halstead_vocabulary",
    "method_halstead_volume",
    "method_halstead_difficulty",
    "method_halstead_effort",
    "method_halstead_bugs",
    "method_halstead_time",
    "method_nargs",
    "method_nexits",
    "method_global_var_count",
    "file_cloc",
    "file_mi",
    "file_nom",
    "file_class",
]
