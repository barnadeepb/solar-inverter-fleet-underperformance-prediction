"""Wait for the Vertex AI AutoML training job launched by
vertex_automl_launch.py to finish, batch-predict on the held-out test set,
and fold the resulting metrics into results/metrics/regression_benchmark.json
under the key "vertex_automl_tables" -- same RMSE/MAE/R2 fields as every
other model, so it slots into the same comparison table.

This blocks until training completes, which can take an hour or more; run
it as a background process.
"""

import json
import pathlib

import numpy as np
import pandas as pd
from google.cloud import aiplatform, storage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import add_time_features, chronological_split, daylight_mask, load_plant

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def _true_test_labels() -> pd.DataFrame:
    """Recompute the same held-out test rows (deterministic pipeline) to
    get ground-truth AC_POWER, joined by the ROW_ID that was uploaded
    alongside the feature-only prediction input."""
    df = load_plant(1)
    _, test_df = chronological_split(df, train_days=24)
    test_day = test_df[daylight_mask(test_df)].reset_index(drop=True)
    test_day = test_day.copy()
    test_day["ROW_ID"] = range(len(test_day))
    return test_day[["ROW_ID", "AC_POWER"]]


def main():
    info_path = RESULTS_DIR / "metrics" / "vertex_job_info.json"
    with open(info_path) as f:
        info = json.load(f)

    aiplatform.init(project=info["project"], location=info["location"], staging_bucket=f"gs://{info['bucket']}")

    job = aiplatform.AutoMLTabularTrainingJob.get(info["training_job_resource_name"])
    print("waiting for training job to complete...")
    job.wait()
    model = job.get_model()
    print(f"training complete, model: {model.resource_name}")

    dest_prefix = f"gs://{info['bucket']}/solar-pv-underperformance/automl/predictions"
    # First attempt hit a transient "machine type temporarily unavailable"
    # capacity error on the default machine type; pin an explicit, widely
    # available machine type to avoid retrying into the same failure.
    batch_job = model.batch_predict(
        job_display_name="solar-plant1-automl-batch-predict",
        gcs_source=info["test_gcs_uri"],
        gcs_destination_prefix=dest_prefix,
        instances_format="csv",
        predictions_format="jsonl",
        machine_type="n1-standard-4",
        starting_replica_count=1,
        max_replica_count=1,
        sync=True,
    )

    storage_client = storage.Client(project=info["project"])
    predictions = []
    for blob in storage_client.list_blobs(
        info["bucket"], prefix=batch_job.output_info.gcs_output_directory.replace(f"gs://{info['bucket']}/", "")
    ):
        if not blob.name.endswith(".jsonl"):
            continue
        for line in blob.download_as_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = int(row["instance"]["ROW_ID"])
            pred_value = float(row["prediction"]["value"])
            predictions.append((row_id, pred_value))

    pred_df = pd.DataFrame(predictions, columns=["ROW_ID", "PREDICTED_AC_POWER"])
    truth_df = _true_test_labels()
    merged = truth_df.merge(pred_df, on="ROW_ID", how="inner")
    print(f"scored {len(merged)} / {len(truth_df)} test rows")

    metrics = {
        "rmse": {"mean": float(np.sqrt(mean_squared_error(merged["AC_POWER"], merged["PREDICTED_AC_POWER"]))), "std": 0.0},
        "mae": {"mean": float(mean_absolute_error(merged["AC_POWER"], merged["PREDICTED_AC_POWER"])), "std": 0.0},
        "r2": {"mean": float(r2_score(merged["AC_POWER"], merged["PREDICTED_AC_POWER"])), "std": 0.0},
        "note": "single run: managed AutoML training is not repeated across seeds",
    }

    benchmark_path = RESULTS_DIR / "metrics" / "regression_benchmark.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    benchmark["results"]["vertex_automl_tables"] = metrics
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"updated {benchmark_path} with vertex_automl_tables: {metrics}")


if __name__ == "__main__":
    main()
