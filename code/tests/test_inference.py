"""CPU regression tests for measurements, routing, local loading, and output files."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from seaquery.inference.conditioning import build_adaptive_prefix, build_bracket_prefix, validate_metrics
from seaquery.inference.measurements import measure_mask
from seaquery.inference.reef_state import health_state, strict_letter, state_from_letter
from seaquery.inference.loaders import adapter_config, compatible_config, resolve_checkpoint, MODEL_REVISIONS
from seaquery.inference.pipeline import Pipeline
from seaquery.inference.run import main

METRICS = {"live": 40, "bleached": 2, "dead": 10, "algae": 5, "cots": 1}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "reef.png"
        Image.new("RGB", (16, 8), color=(12, 98, 45)).save(self.image)

    def test_foreground_denominator(self):
        mask = np.array([[0, 13, 6, 4], [3, 10, 1, 38]], dtype=np.uint8)
        result = measure_mask(mask)
        self.assertAlmostEqual(result["live"], 100 / 6)
        self.assertEqual(result["cots"], 0)

    def test_cots_area_threshold_and_connectivity(self):
        mask = np.ones((30, 30), dtype=np.uint8)
        mask[0:10, 0:10] = 38  # exactly 100 is excluded
        mask[11:22, 11:21] = 38  # 110 counted
        self.assertEqual(measure_mask(mask)["cots"], 1)
        mask[:] = 1
        mask[:10, :10] = 38
        mask[10:20, 10:20] = 38  # diagonal touch must not join components
        self.assertEqual(measure_mask(mask)["cots"], 0)

    def test_invalid_mask_rejected(self):
        for mask in (np.zeros((2, 2, 3), dtype=np.uint8), np.array([[40]]), np.array([[1.2]])):
            with self.assertRaises(ValueError):
                measure_mask(mask)

    def test_rule_boundaries(self):
        for live, bleach, state in [(30, 0, "stressed"), (31, 10, "stressed"),
                                    (10, 0, "stressed"), (9, 0, "degraded"),
                                    (40, 25, "stressed"), (40, 26, "degraded"), (31, 9, "healthy")]:
            self.assertEqual(health_state(live, bleach), state)

    def test_strict_parser_and_permuted_options(self):
        self.assertEqual(strict_letter(" b "), "B")
        for text in ("B.", "Answer: B", "healthy", "", None):
            self.assertIsNone(strict_letter(text))
        mapping = dict(option_A_state="degraded", option_B_state="healthy", option_C_state="stressed")
        self.assertEqual(state_from_letter(mapping, "B"), "healthy")

    def test_conditioning_text_and_empty_adaptive(self):
        self.assertEqual(build_adaptive_prefix(METRICS, "What color is the fish?"), "")
        self.assertEqual(build_adaptive_prefix(METRICS, "Is there bleaching?"), "[Reef metrics - Bleached coral: 2.0%] ")
        self.assertIn("COTS: 1]", build_bracket_prefix(METRICS))
        self.assertEqual(build_adaptive_prefix(METRICS, "What is the reef health?"), build_bracket_prefix(METRICS))

    def test_invalid_measurements_rejected(self):
        for update in ({"live": 101}, {"cots": 1.5}, {"bleached": float("nan")}, {"dead": -1}):
            with self.assertRaises(ValueError):
                validate_metrics({**METRICS, **update})

    def test_adapter_base_mismatch(self):
        (self.root / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "wrong"}))
        with self.assertRaisesRegex(ValueError, "selected base"):
            adapter_config(self.root, "expected")
        self.assertEqual(compatible_config({"r": 8, "use_qalora": False}, {"r"}), {"r": 8})
        with self.assertRaises(ValueError):
            compatible_config({"r": 8, "use_qalora": True}, {"r"})

    def test_adapter_alias_mismatch(self):
        with self.assertRaises(ValueError):
            Pipeline(self.root, model="llava", adapter="internvl-lora")

    def test_on_demand_resolution_pins_revision_and_cache(self):
        source, options, metadata = resolve_checkpoint("llava", self.root, self.root / "cache")
        self.assertEqual(source, "llava-hf/llava-1.5-7b-hf")
        self.assertEqual(options["revision"], MODEL_REVISIONS["llava"])
        self.assertFalse(options["local_files_only"])
        self.assertEqual(metadata["location"], "hub_cache")
        self.assertTrue(resolve_checkpoint("llava", self.root, self.root / "cache", offline=True)[1]["local_files_only"])

    def test_partial_local_snapshot_does_not_silently_download(self):
        folder = self.root / "llava"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint("llava", self.root, self.root / "cache")

    def test_segment_existing_mask_and_no_overwrite(self):
        mask = self.root / "mask.npy"
        np.save(mask, np.full((8, 16), 6, dtype=np.uint8))
        output = self.root / "results"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["segment", "--image", str(self.image), "--mask", str(mask), "--output", str(output)])
        result = json.loads((output / "metrics.json").read_text())
        self.assertEqual(result["metrics"]["live"], 100)
        self.assertEqual(result["reef_state"], "healthy")
        self.assertTrue((output / "mask.npy").exists())
        with self.assertRaises(FileExistsError):
            Pipeline(self.root).measure(self.image, output, mask)

    def test_empty_foreground_not_classified(self):
        mask = self.root / "empty.npy"
        np.save(mask, np.zeros((8, 16), dtype=np.uint8))
        result = Pipeline(self.root).measure(self.image, mask_path=mask)
        self.assertIsNone(result["reef_state"])

    def test_invalid_option_mapping_rejected_before_loading(self):
        pipeline = Pipeline(self.root, model="llava")
        with self.assertRaisesRegex(ValueError, "each reef state"):
            pipeline.answer(self.image, "Question", option_order=["healthy"] * 3)

    def test_missing_metrics_rejected_before_loading(self):
        pipeline = Pipeline(self.root, model="llava")
        with self.assertRaisesRegex(ValueError, "Conditioning requires"):
            pipeline.answer(self.image, "Question", mode="prefix-full")

    def test_batch_writes_per_mode_records(self):
        path = self.root / "batch.csv"
        path.write_text('question_id,image,question\nq1,reef.png,Is there bleaching?\n')
        out = self.root / "predictions.jsonl"
        with patch.object(Pipeline, "answer", return_value={"answer": "No"}) as answer:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["batch", "--input", str(path), "--images-root", str(self.root),
                      "--model", "llava", "--modes", "none", "prefix-full", "--auto-segment", "--output", str(out)])
        self.assertEqual(answer.call_count, 2)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question_id"], "q1")


if __name__ == "__main__":
    unittest.main()
