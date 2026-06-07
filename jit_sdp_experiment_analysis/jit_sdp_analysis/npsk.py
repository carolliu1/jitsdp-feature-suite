from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import pandas as pd

from .config import NPSKConfig


class ScottKnottESD:
    """Strict adapter for the R package from klainfo/ScottKnottESD.

    This class intentionally delegates ranking to ScottKnottESD::sk_esd() via
    Rscript. The project does not include a local Python approximation.
    """

    def __init__(self, config: NPSKConfig | None = None) -> None:
        self.config = config or NPSKConfig()

    def rank(self, matrix: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
        numeric = matrix.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(axis=1, how="all")
        if numeric.empty:
            return pd.DataFrame(columns=["concept", "raw_group", "rank", "cluster"])

        ranking_input = numeric if higher_is_better else -numeric
        result = self._rank_with_r_package(ranking_input)
        result["cluster"] = result["rank"]
        return result.sort_values(["rank", "concept"]).reset_index(drop=True)

    def _rank_with_r_package(self, matrix: pd.DataFrame) -> pd.DataFrame:
        script_path = Path(__file__).resolve().parent / "r" / "run_scottknottesd.R"
        if not script_path.exists():
            raise FileNotFoundError(f"ScottKnottESD R runner is missing: {script_path}")

        with tempfile.TemporaryDirectory(prefix="sk_esd_", dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.csv"
            output_path = temp_path / "output.csv"
            matrix.to_csv(input_path, index=False)

            command = [
                self.config.rscript_path,
                str(script_path),
                str(input_path),
                str(output_path),
                self.config.r_version,
                str(self.config.alpha),
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Strict ScottKnottESD backend requires Rscript, but Rscript was not found. "
                    "Install R and the GitHub package with: "
                    'install.packages("remotes"); '
                    'remotes::install_github("klainfo/ScottKnottESD", ref="development")'
                ) from exc

            if completed.returncode != 0:
                raise RuntimeError(
                    "ScottKnottESD::sk_esd failed. stderr:\n"
                    f"{completed.stderr}\nstdout:\n{completed.stdout}"
                )

            return pd.read_csv(output_path)

