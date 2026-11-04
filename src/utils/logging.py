"""Experiment logging: MLflow plus a plain CSV per run.

MLflow's tracking URI comes from ``MLFLOW_TRACKING_URI``; if unset, the local folder store
``./mlruns`` is used. MLflow 3 refuses the folder store unless ``MLFLOW_ALLOW_FILE_STORE`` is set,
which this module does. The folder store keeps one directory per run, so concurrent cluster jobs
never contend for a shared database file (SQLite locking is unreliable on Lustre).

The CSV (``metrics.csv`` in the run's output directory) is a dependency-free record of the same
metrics, appended as training goes, so a resumed run continues it.
"""

import csv
import os
from pathlib import Path

import mlflow

from src.utils.paths import REPO_ROOT


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


class RunLogger:
    """Logs metrics to MLflow and to ``<out_dir>/metrics.csv``; resumes an existing run."""

    def __init__(
        self, experiment: str, run_name: str, out_dir: Path, config: dict, run_id: str | None = None
    ) -> None:
        if "MLFLOW_TRACKING_URI" not in os.environ:
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri((REPO_ROOT / "mlruns").as_uri())
        mlflow.set_experiment(experiment)
        if run_id is None:
            self.run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({k: str(v)[:500] for k, v in _flatten(config).items()})
        else:
            self.run = mlflow.start_run(run_id=run_id)
        self.run_id = self.run.info.run_id
        self.csv_path = Path(out_dir) / "metrics.csv"

    def log(self, metrics: dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)
        new_file = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["step", "metric", "value"])
            writer.writerows([step, k, v] for k, v in metrics.items())

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def close(self) -> None:
        mlflow.end_run()
