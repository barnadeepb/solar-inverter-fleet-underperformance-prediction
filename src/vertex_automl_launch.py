"""Launch a Vertex AI AutoML Tables regression job as the "proprietary
managed-service" baseline called for by the rigor checklist -- comparing
the hand-built models above against what a practitioner gets by pointing a
commercial AutoML product at the same train/test split.

This only launches the job (sync=False) and records its resource name;
AutoML Tables training takes on the order of an hour or more even on a
small dataset, so waiting for it here would block everything else. Run
vertex_automl_evaluate.py afterward (separately) to fetch the trained
model, batch-predict on the held-out test set, and fold the metrics into
results/metrics/regression_benchmark.json.

Cost note: this creates real billable Vertex AI training compute on the
configured GCP project (minimum budget used below: 1 node-hour).
"""

import json
import pathlib

from google.cloud import aiplatform, storage

from data_prep import add_time_features, chronological_split, daylight_mask, load_plant

PROJECT = "project-0615c873-134d-4b53-b2e"
LOCATION = "us-central1"
BUCKET = "cloud-ai-platform-044fb91d-44b7-4006-b40d-6ba11e1b2e0c"
GCS_PREFIX = "solar-pv-underperformance/automl"

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
FEATURE_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "HOUR_SIN", "HOUR_COS"]
TARGET_COL = "AC_POWER"


def _write_and_upload_csv(df, columns, local_path: pathlib.Path, gcs_name: str, bucket) -> str:
    df[columns].to_csv(local_path, index=False)
    blob = bucket.blob(f"{GCS_PREFIX}/{gcs_name}")
    blob.upload_from_filename(str(local_path))
    return f"gs://{BUCKET}/{GCS_PREFIX}/{gcs_name}"


def main():
    df = load_plant(1)
    train_df, test_df = chronological_split(df, train_days=24)
    train_day = train_df[daylight_mask(train_df)].reset_index(drop=True)
    test_day = test_df[daylight_mask(test_df)].reset_index(drop=True)
    test_day = test_day.copy()
    test_day["ROW_ID"] = range(len(test_day))

    local_dir = RESULTS_DIR / "vertex_data"
    local_dir.mkdir(parents=True, exist_ok=True)

    storage_client = storage.Client(project=PROJECT)
    bucket = storage_client.bucket(BUCKET)

    train_uri = _write_and_upload_csv(
        train_day, FEATURE_COLS + [TARGET_COL], local_dir / "train.csv", "train.csv", bucket
    )
    # Target column deliberately excluded from the prediction input -- only
    # ROW_ID travels through so predictions can be joined back to the
    # (locally recomputed, deterministic) ground truth for scoring.
    test_uri = _write_and_upload_csv(
        test_day, FEATURE_COLS + ["ROW_ID"], local_dir / "test_features.csv", "test_features.csv", bucket
    )
    print(f"uploaded train -> {train_uri}")
    print(f"uploaded test  -> {test_uri}")

    aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=f"gs://{BUCKET}")

    dataset = aiplatform.TabularDataset.create(
        display_name="solar-plant1-ac-power-train",
        gcs_source=train_uri,
    )

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="solar-plant1-ac-power-automl",
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse",
    )

    model = job.run(
        dataset=dataset,
        target_column=TARGET_COL,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=1000,
        model_display_name="solar-plant1-ac-power-automl-model",
        disable_early_stopping=False,
        sync=False,
    )
    job.wait_for_resource_creation()

    info = {
        "project": PROJECT,
        "location": LOCATION,
        "bucket": BUCKET,
        "train_gcs_uri": train_uri,
        "test_gcs_uri": test_uri,
        "training_job_resource_name": job.resource_name,
    }
    out_path = RESULTS_DIR / "metrics" / "vertex_job_info.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"launched training job: {job.resource_name}")
    print(f"wrote {out_path}")
    print("run vertex_automl_evaluate.py later (this can take an hour or more) to score it")


if __name__ == "__main__":
    main()
