"""Rank a CSV uploaded to the deployment bucket's incoming/ prefix."""

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote_plus

try:
    import boto3
except ImportError:  # Allows local unit tests to inject a fake S3 client.
    boto3 = None

from processing.scouting_channel import run_scouting_pipeline
from shared.cloud_infra import DistributedStorage, ElasticComputePool


def lambda_handler(event, context, s3_client=None, work_dir=None):
    """S3 event -> rank, select the best-supported method, and publish reports."""
    client = s3_client or (boto3.client("s3") if boto3 else None)
    if client is None:
        raise RuntimeError("boto3 is required in AWS Lambda; inject s3_client for local tests.")
    processed = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        if not key.startswith("incoming/") or not key.lower().endswith(".csv"):
            continue
        with tempfile.TemporaryDirectory(prefix="scouting-", dir=work_dir) as tmp:
            root = Path(tmp)
            input_path, output_dir = root / Path(key).name, root / "results"
            response = client.get_object(Bucket=bucket, Key=key)
            with input_path.open("wb") as handle:
                handle.write(response["Body"].read())

            compute = ElasticComputePool(min_workers=1, max_workers=4)
            try:
                summary = run_scouting_pipeline(
                    compute, DistributedStorage(), dataset_path=input_path,
                    min_minutes=float(os.environ.get("MIN_MINUTES", "180")),
                    output_dir=output_dir,
                )
            finally:
                compute.shutdown()

            workbook_path = Path(summary["rankings_workbook_file"])
            workbook_key = f"results/{workbook_path.name}"
            client.upload_file(str(workbook_path), bucket, workbook_key,
                               ExtraArgs={"ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})
            result = {
                "input_bucket": bucket, "input_key": key,
                "output_bucket": bucket, "workbook_key": workbook_key,
                "selected_algorithm": summary["selected_algorithm_label"],
                "players": summary["total_players_scouted"], "algorithms": summary["algorithms"],
            }
            print(json.dumps(result, sort_keys=True))
            processed.append(result)
    return {"processed_files": len(processed), "results": processed}
