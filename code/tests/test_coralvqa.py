"""Tests of the repository's frozen evaluation schema and CLI integration."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

from seaquery.inference.coralvqa import CoralVQA
from seaquery.inference.coralscapes import CoralScapes
from seaquery.inference.paths import require_local_file
from seaquery.inference.pipeline import Pipeline
from seaquery.inference.run import main, parser


class CoralVQATests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        Image.new("RGB", (8, 4)).save(self.root / "images" / "reef.jpg")
        self.rows = [{"id": "CoralVQA", "image": "reef.jpg",
                      "question": "<image>\n[vqa] Is there coral?  Provide your answer as short as possible.",
                      "answer": "Yes", "basic_visual_label": "Coral Presence",
                      "ecological_label": "Coral Habitat Environment", "question_clean": "is there coral?",
                      "eval_row_id": f"coralvqa_eval_{i:05d}", "pair_occurrence": i} for i in range(2)]
        (self.root / "coralvqa_split_imagegrouped.json").write_text(json.dumps({
            "train_images": ["not_downloaded.jpg"], "val_images": ["reef.jpg", "unused.jpg"]}))
        self.write_index()
        image = self.root / "images" / "reef.jpg"
        with (self.root / "image_manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "bytes", "sha256"])
            writer.writeheader()
            writer.writerow({"image": "reef.jpg", "bytes": image.stat().st_size,
                             "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})

    def write_index(self):
        index = self.root / "val_index.json"
        index.write_text(json.dumps(self.rows))
        (self.root / "subset_manifest.json").write_text(json.dumps({
            "scope": "evaluation_only", "questions": 2, "images": 1,
            "source_sha256": {"val_index.json": hashlib.sha256(index.read_bytes()).hexdigest()}}))

    def test_load_preserves_duplicates_ids_and_reference_text(self):
        ds = CoralVQA(self.root)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.records, self.rows)
        self.assertEqual(ds.select(1, 1)[0]["eval_row_id"], "coralvqa_eval_00001")
        self.assertEqual(ds.validate_images(verify_hashes=True), 1)
        self.assertEqual(ds.metadata(self.rows[0], 0)["reference_answer"], "Yes")

    def test_csv_and_jsonl_equivalence_and_mismatch(self):
        for suffix in ("csv", "jsonl"):
            p = self.root / f"coralvqa_eval.{suffix}"
            if suffix == "csv":
                fields = ["question_id", "image", "question", "answer", "basic_visual_label", "ecological_label"]
                with p.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for row in self.rows:
                        writer.writerow({k: row["eval_row_id"] if k == "question_id" else row[k] for k in fields})
            else:
                p.write_text("\n".join(json.dumps(r) for r in self.rows))
            self.assertEqual(CoralVQA(self.root, p).records, self.rows)
            p.write_text(p.read_text().replace('Yes', 'No'))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                CoralVQA(self.root, p)

    def test_frozen_index_tampering_and_training_leakage(self):
        p = self.root / "val_index.json"
        p.write_text(p.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            CoralVQA(self.root)
        self.write_index()
        (self.root / "coralvqa_split_imagegrouped.json").write_text(json.dumps({"train_images": ["reef.jpg"], "val_images": ["reef.jpg"]}))
        with self.assertRaisesRegex(ValueError, "split"):
            CoralVQA(self.root)

    def test_missing_image_and_lfs_pointer(self):
        image = self.root / "images" / "reef.jpg"
        image.unlink()
        with self.assertRaises(FileNotFoundError):
            CoralVQA(self.root).validate_images()
        image.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
        with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
            CoralVQA(self.root).validate_images()
        shard = self.root / "test-00000-of-00001.parquet"
        shard.write_bytes(image.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
            CoralScapes(self.root)

    def test_bad_ids_and_path_rejected(self):
        self.rows[1]["eval_row_id"] = self.rows[0]["eval_row_id"]
        self.write_index()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            CoralVQA(self.root)
        self.rows[1]["eval_row_id"] = "second"
        self.rows[1]["image"] = "../reef.jpg"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "filename"):
            CoralVQA(self.root)

    def test_hash_verification_catches_same_size_change(self):
        image = self.root / "images" / "reef.jpg"
        b = bytearray(image.read_bytes())
        b[-1] ^= 1
        image.write_bytes(b)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            CoralVQA(self.root).validate_images(verify_hashes=True)

    def test_cli_retains_references_but_does_not_send_them_to_model(self):
        seen = []
        def predict(state, image, question, **kwargs):
            seen.append(question)
            self.assertNotIn("<image>", question)
            self.assertNotIn("[vqa]", question)
            self.assertNotIn("Yes", question)
            return "No"
        state = SimpleNamespace(device=SimpleNamespace(type="cpu"), metadata={})
        out = self.root / "predictions.jsonl"
        with patch.object(Pipeline, "initialize_vlm", return_value=state), \
             patch("seaquery.inference.llava.predict", side_effect=predict), \
             contextlib.redirect_stdout(io.StringIO()):
            main(["coralvqa", "--coralvqa-root", str(self.root), "--model", "llava", "--output", str(out)])
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(seen), 2)
        self.assertEqual([r["question_id"] for r in rows], [r["eval_row_id"] for r in self.rows])
        self.assertEqual(rows[0]["answer"], "No")
        self.assertEqual(rows[0]["reference_answer"], "Yes")
        self.assertEqual(rows[0]["question_raw"], self.rows[0]["question"])
        self.assertIn("Provide your answer as short as possible.", rows[0]["question"])

    def test_default_repo_paths(self):
        args = parser().parse_args(["coralvqa", "--model", "llava"])
        self.assertEqual(args.coralvqa_root.parts[-3:], ("data", "datasets", "coralvqa"))
        args = parser().parse_args(["coralscapes-info", "--split", "test"])
        self.assertEqual(args.coralscapes_root.parts[-3:], ("data", "datasets", "coralscapes"))
        self.assertEqual(args.coralscapes_root.parent.parent.parent, Path(__file__).resolve().parents[2])

    def test_missing_conditioning_metrics_and_limits(self):
        ds = CoralVQA(self.root)
        for start, limit in ((-1, 1), (2, 1), (0, 0)):
            with self.assertRaises(ValueError):
                ds.select(start, limit)
        with self.assertRaisesRegex(ValueError, "requires --auto-segment"):
            main(["coralvqa", "--coralvqa-root", str(self.root), "--model", "llava",
                  "--modes", "prefix-adaptive", "--output", str(self.root / "p.jsonl")])


if __name__ == "__main__":
    unittest.main()
