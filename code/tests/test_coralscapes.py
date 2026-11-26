"""Small real Parquet fixtures; no checkpoint downloads or GPU required."""
import contextlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from seaquery.inference.coralscapes import CoralScapes
from seaquery.inference._common import open_rgb
from seaquery.inference.pipeline import Pipeline, sha256_file
from seaquery.inference.run import main


def encoded(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ParquetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.rows = []
        for i in range(6):
            mask = Image.fromarray(np.full((4, 8), 6 + i, dtype=np.uint8)).convert("P")
            self.rows.append({"image": {"bytes": encoded(Image.new("RGB", (8, 4), (i, 50, 100))),
                                        "path": f"original/{i}.jpg"},
                              "label": {"bytes": encoded(mask), "path": f"labels/{i}.png"}})
        for shard in range(2):
            pq.write_table(pa.Table.from_pylist(self.rows[shard * 3:(shard + 1) * 3]),
                           self.data / f"test-{shard:05d}-of-00002.parquet", row_group_size=2)

    def test_shard_and_row_group_selection(self):
        dataset = CoralScapes(self.root)
        self.assertEqual(len(dataset), 6)
        for i in range(6):
            sample, mask = dataset.get(i, include_label=True)
            self.assertEqual(open_rgb(sample).getpixel((0, 0)), (i, 50, 100))
            np.testing.assert_array_equal(mask, np.full((4, 8), 6 + i))
            self.assertEqual(sample.provenance["shard_row"], i % 3)
            self.assertEqual(sample.provenance["row_group"], (i % 3) // 2)
            self.assertEqual(sample.provenance["original_image_path"], f"original/{i}.jpg")
        self.assertEqual(len(CoralScapes(self.data)), 6)
        self.assertEqual(sorted(p.name for p in self.data.iterdir()),
                         ["test-00000-of-00002.parquet", "test-00001-of-00002.parquet"])

    def test_incomplete_shards_and_invalid_indices(self):
        ds = CoralScapes(self.root)
        for index in (-1, 6, True, "0"):
            with self.assertRaises(IndexError):
                ds.get(index)
        (self.data / "test-00000-of-00002.parquet").unlink()
        with self.assertRaises(ValueError):
            CoralScapes(self.root)
        with self.assertRaises(FileNotFoundError):
            CoralScapes(self.root, "train")

    def test_default_does_not_decode_reference(self):
        # A broken annotation must not affect image-only inference.
        self.rows[0]["label"]["bytes"] = b"not a mask"
        pq.write_table(pa.Table.from_pylist(self.rows[:3]), self.data / "test-00000-of-00002.parquet")
        sample, label = CoralScapes(self.root).get(0)
        self.assertIsNone(label)
        self.assertEqual(sample.rgb().size, (8, 4))
        with self.assertRaises(Exception):
            CoralScapes(self.root).get(0, include_label=True)

    def test_export_and_no_overwrite(self):
        out = self.root / "export"
        args = ["coralscapes-export", "--coralscapes-root", str(self.root), "--split", "test",
                "--index", "4", "--include-reference-mask", "--output", str(out)]
        with contextlib.redirect_stdout(StringIO()):
            main(args)
        np.testing.assert_array_equal(np.load(out / "reference_mask.npy"), np.full((4, 8), 10))
        self.assertEqual(json.loads((out / "source.json").read_text())["split_index"], 4)
        with self.assertRaises(FileExistsError):
            main(args)

    def test_segment_uses_predictions_and_retains_provenance(self):
        out = self.root / "segment"
        with patch("seaquery.inference.pipeline.Segmentor") as model:
            model.return_value.predict.return_value = np.full((4, 8), 3, dtype=np.uint8)
            model.return_value.source = {"location": "test"}
            with contextlib.redirect_stdout(StringIO()):
                main(["segment", "--coralscapes-root", str(self.root), "--split", "test",
                      "--index", "0", "--output", str(out)])
            self.assertEqual(model.return_value.predict.call_count, 1)
        result = json.loads((out / "metrics.json").read_text())
        self.assertEqual(result["metrics"]["dead"], 100)
        self.assertEqual(result["metrics"]["live"], 0)
        self.assertEqual(result["image_source"]["split_index"], 0)
        self.assertEqual(result["image_sha256"], CoralScapes(self.root).get(0)[0].sha256)

    def test_vqa_and_batch_accept_parquet_images(self):
        def predict(state, image, question, **kw):
            self.assertEqual(open_rgb(image).size, (8, 4))
            return "Yes"
        fake = SimpleNamespace(device=SimpleNamespace(type="cpu"), metadata={})
        with patch.object(Pipeline, "initialize_vlm", return_value=fake), \
             patch("seaquery.inference.llava.predict", side_effect=predict):
            out = self.root / "vqa.json"
            with contextlib.redirect_stdout(StringIO()):
                main(["vqa", "--model", "llava", "--coralscapes-root", str(self.root), "--split", "test",
                      "--index", "5", "--question", "Is there coral?", "--output", str(out)])
            result = json.loads(out.read_text())[0]
            self.assertEqual(result["answer"], "Yes")
            self.assertEqual(result["image_source"]["split_index"], 5)
            table = self.root / "questions.csv"
            table.write_text('question_id,split,index,question\na,test,0,Any coral?\nb,test,0,Any fish?\nc,test,3,Any coral?\n')
            out = self.root / "batch.jsonl"
            with contextlib.redirect_stdout(StringIO()):
                main(["batch", "--model", "llava", "--coralscapes-root", str(self.root),
                      "--input", str(table), "--output", str(out)])
            results = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([r["image_source"]["split_index"] for r in results], [0, 0, 3])

    def test_missing_embedded_bytes_and_rgb_label_fail(self):
        path = self.data / "test-00000-of-00002.parquet"
        self.rows[0]["image"]["bytes"] = None
        pq.write_table(pa.Table.from_pylist(self.rows[:3]), path)
        with self.assertRaisesRegex(ValueError, "embedded image bytes"):
            CoralScapes(self.root).get(0)
        self.rows[0]["image"]["bytes"] = self.rows[1]["image"]["bytes"]
        self.rows[0]["label"]["bytes"] = encoded(Image.new("RGB", (8, 4)))
        pq.write_table(pa.Table.from_pylist(self.rows[:3]), path)
        with self.assertRaisesRegex(ValueError, "2-D integer"):
            CoralScapes(self.root).get(0, include_label=True)


if __name__ == "__main__":
    unittest.main()
