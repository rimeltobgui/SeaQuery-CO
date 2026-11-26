"""Connect local segmentation, measurements, metric conditioning, and VLMs."""
import hashlib
import importlib
import json
from pathlib import Path
import time
import numpy as np

from .paths import require_local_file
from .coralscapes import ParquetImage
from .conditioning import build_adaptive_prefix, build_bracket_prefix, validate_metrics, normalize_question
from .loaders import ADAPTERS, load_vlm
from .measurements import measure_mask
from .reef_state import health_state, strict_letter, state_from_letter, STATES
from .segmentation import PROTOCOL, Segmentor

MODES = ("none", "prefix-adaptive", "prefix-full", "fusion-adaptive", "fusion-full")


def sha256_file(path):
    if isinstance(path, ParquetImage):
        return path.sha256
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_metrics(path):
    value = json.loads(Path(path).read_text())
    # Accept either a raw measurement JSON or this package's segment output.
    return validate_metrics(value.get("metrics", value))


class Pipeline:
    def __init__(self, models_root, model=None, adapter=None, device="auto", precision="auto",
                 quantization="none", segmentor_device="cpu", max_tiles=12, max_new_tokens=None,
                 cache_dir=None, offline=False):
        self.models_root = Path(models_root).resolve()
        self.name = model
        self.adapter = adapter
        self.device, self.precision, self.quantization = device, precision, quantization
        self.segmentor_device, self.max_tiles = segmentor_device, max_tiles
        self.max_new_tokens = max_new_tokens
        self.cache_dir, self.offline = cache_dir, offline
        if not 1 <= max_tiles <= 12:
            raise ValueError("max_tiles must be between 1 and 12")
        if max_new_tokens is not None and max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if adapter and str(adapter) in ADAPTERS:
            if ADAPTERS[str(adapter)] != model:
                raise ValueError("Adapter alias does not match selected model")
            self.adapter = self.models_root / "adapters" / str(adapter)
        self.state = None
        self.segmentor = None
        self.measurement_cache = {}

    def initialize_vlm(self):
        if self.state is None:
            if self.name is None:
                raise ValueError("Select a VLM")
            self.state = load_vlm(self.name, self.models_root, self.adapter, self.device,
                                  self.precision, self.quantization, self.cache_dir, self.offline)
        return self.state

    def measure(self, image, output_dir=None, mask_path=None):
        if not isinstance(image, ParquetImage):
            image = Path(image).resolve()
            require_local_file(image)
        key = (image.name if isinstance(image, ParquetImage) else str(image),
               sha256_file(image), str(mask_path) if mask_path else None)
        if key in self.measurement_cache and output_dir is None:
            return self.measurement_cache[key]
        if mask_path:
            mask = np.load(mask_path, allow_pickle=False)
            source = "supplied_label_mask"
        else:
            if self.segmentor is None:
                self.segmentor = Segmentor(self.models_root, self.segmentor_device, self.cache_dir, self.offline)
            mask = self.segmentor.predict(image)
            source = "segformer-b2"
        metrics = validate_metrics(measure_mask(mask))
        foreground = int(np.count_nonzero((mask != 0) & (mask != 13)))
        result = {"image": image.name if isinstance(image, ParquetImage) else str(image), "image_sha256": sha256_file(image), "metrics": metrics,
                  "foreground_pixels": foreground, "mask_shape": list(mask.shape),
                  "reef_state": health_state(metrics["live"], metrics["bleached"]) if foreground else None,
                  "measurement_source": source,
                  "protocol": {**PROTOCOL, "model": source},
                  "warning": "No foreground pixels; reef state is undefined" if not foreground else None}
        if isinstance(image, ParquetImage):
            result["image_source"] = image.provenance
        if mask_path:
            result["input_mask"] = str(Path(mask_path).resolve())
            result["input_mask_sha256"] = sha256_file(mask_path)
        else:
            result["checkpoint_source"] = self.segmentor.source
        if output_dir:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            for name in ("mask.npy", "metrics.json"):
                if (output / name).exists():
                    raise FileExistsError(f"Refusing to overwrite {output / name}")
            with (output / "mask.npy").open("wb") as stream:
                np.save(stream, mask)
            result["mask_path"] = str((output / "mask.npy").resolve())
            (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        self.measurement_cache[key] = result
        return result

    def answer(self, image, question, mode="none", metrics=None, auto_segment=False, option_order=None):
        if mode not in MODES:
            raise ValueError(f"Unknown conditioning mode: {mode}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty text")
        if not isinstance(image, ParquetImage):
            image = Path(image).resolve()
            require_local_file(image)
        question_raw = question
        question = normalize_question(question)
        if not question:
            raise ValueError("Question is empty after removing dataset tags")
        if option_order is not None and (len(option_order) != 3 or set(option_order) != set(STATES)):
            raise ValueError("option_order must contain each reef state exactly once")
        injected = ""
        origin = None
        if mode != "none":
            if metrics is None:
                if not auto_segment:
                    raise ValueError("Conditioning requires measurements or --auto-segment")
                measurement = self.measure(image)
                if not measurement["foreground_pixels"]:
                    raise ValueError("Cannot condition on an image with no segmented foreground")
                metrics = measurement["metrics"]
                origin = "segformer-b2"
            else:
                origin = "supplied_metrics"
            metrics = validate_metrics(metrics)
            builder = build_adaptive_prefix if mode.endswith("adaptive") else build_bracket_prefix
            injected = builder(metrics, question)
        state = self.initialize_vlm()
        backend = "paligemma" if self.name.startswith("paligemma") else self.name
        module = importlib.import_module(f".{backend}", __package__)
        options = {"injected": injected, "fusion": mode.startswith("fusion")}
        defaults = {"internvl": 16, "blip2": 4}
        options["max_new_tokens"] = self.max_new_tokens or defaults.get(self.name, 32)
        if self.name == "internvl":
            options["max_tiles"] = self.max_tiles
        import torch
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        raw = module.predict(state, image, question, **options)
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result = {"image": image.name if isinstance(image, ParquetImage) else str(image), "image_sha256": sha256_file(image), "question": question,
                  "question_raw": question_raw, "mode": mode, "metrics": metrics if mode != "none" else None,
                  "measurement_source": origin, "metric_text": injected, "answer": raw,
                  "prediction_seconds": elapsed,
                  "generation": {"max_new_tokens": options["max_new_tokens"], "do_sample": False,
                                 "internvl_max_tiles": self.max_tiles if self.name == "internvl" else None},
                  "runtime": state.metadata}
        if isinstance(image, ParquetImage):
            result["image_source"] = image.provenance
        if option_order is not None:
            mapping = {f"option_{letter}_state": value for letter, value in zip("ABC", option_order)}
            letter = strict_letter(raw)
            result.update({"option_mapping": mapping, "parsed_letter": letter,
                           "parsed_reef_state": state_from_letter(mapping, letter)})
        return result
