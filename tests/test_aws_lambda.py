import csv
import io
import tempfile
import unittest
from pathlib import Path

from aws.lambda_function import lambda_handler


class FakeS3:
    def __init__(self, payload):
        self.payload = payload.encode("utf-8")
        self.uploads = []

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.payload)}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        with open(Filename, "rb") as handle:
            self.uploads.append((Bucket, Key, handle.read(), ExtraArgs))


class AwsRankerHandlerTest(unittest.TestCase):
    def test_s3_upload_event_generates_one_consolidated_excel_report(self):
        headers = ["player_id", "name", "position", "team", "country", "minutes", "goals_per90",
                   "assists_per90", "xg_per90", "shots_on_target_rate", "season", "league", "source"]
        rows = [
            ["one", "Forward One", "FWD", "Club", "A", "900", "0.8", "0.2", "0.7", "0.5",
             "2022-2023", "League", "research export"],
            ["two", "Forward Two", "FWD", "Club", "A", "900", "0.3", "0.1", "0.4", "0.4",
             "2022-2023", "League", "research export"],
        ]
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(headers)
        writer.writerows(rows)
        fake_s3 = FakeS3(buffer.getvalue())
        event = {"Records": [{"s3": {"bucket": {"name": "test-bucket"},
                                     "object": {"key": "incoming/player_season.csv"}}}]}

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as work_dir:
            result = lambda_handler(event, None, s3_client=fake_s3, work_dir=work_dir)

        self.assertEqual(result["processed_files"], 1)
        self.assertEqual(len(fake_s3.uploads), 1)
        uploads = {key: (bucket, payload, metadata)
                   for bucket, key, payload, metadata in fake_s3.uploads}
        bucket, payload, metadata = uploads["results/FINAL_TALENT_SCOUTING_RESULTS.xlsx"]
        self.assertEqual(bucket, "test-bucket")
        self.assertTrue(payload.startswith(b"PK"))
        self.assertEqual(metadata["ContentType"],
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    unittest.main()
