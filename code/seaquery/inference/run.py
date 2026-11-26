"""Run segmentation or VQA with on-demand base models and local adapters."""
import argparse
import csv
import json
import os
from pathlib import Path

from .paths import MODELS_ROOT, RESULTS_ROOT, CORALSCAPES_ROOT, CORALVQA_ROOT, new_run_directory


def defaults():
    return MODELS_ROOT, RESULTS_ROOT


def image_arguments(command):
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--coralscapes-root", type=Path, nargs="?", const=CORALSCAPES_ROOT, help="CoralScapes root or its data/ shard directory")
    command.add_argument("--split", choices=["train", "validation", "test"])
    command.add_argument("--index", type=int, help="Zero-based row within the selected complete split")


def selected_image(args):
    if args.coralscapes_root is not None:
        if args.split is None or args.index is None:
            raise ValueError("Parquet input requires --split and --index")
        from .coralscapes import CoralScapes
        return CoralScapes(args.coralscapes_root, args.split).get(args.index)[0]
    if args.split is not None or args.index is not None:
        raise ValueError("--split/--index require --coralscapes-root")
    return args.image


def parser():
    from .loaders import VLM_NAMES
    from .pipeline import MODES
    models, results = defaults()
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    segment = commands.add_parser("segment", help="Save a class mask and ecological measurements")
    image_arguments(segment)
    segment.add_argument("--models-root", type=Path, default=models)
    segment.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    segment.add_argument("--mask", type=Path, help="Measure an existing integer .npy mask instead of running SegFormer")
    segment.add_argument("--output", type=Path, help="Output directory; default: a unique run folder under results/segment/")
    segment.add_argument("--cache-dir", type=Path, help="Temporary/cache directory for selected base checkpoints")
    segment.add_argument("--offline", action="store_true", help="Require already cached or local checkpoints")
    for name in ("vqa", "batch", "coralvqa"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--model", choices=VLM_NAMES, required=True)
        cmd.add_argument("--models-root", type=Path, default=models)
        cmd.add_argument("--cache-dir", type=Path, help="Temporary/cache directory for selected base checkpoints")
        cmd.add_argument("--offline", action="store_true", help="Require already cached or local checkpoints")
        cmd.add_argument("--adapter", help="Adapter folder path or registered alias")
        cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        cmd.add_argument("--precision", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
        cmd.add_argument("--quantization", choices=["none", "4bit"], default="none")
        cmd.add_argument("--modes", nargs="+", choices=MODES, default=["none"])
        cmd.add_argument("--auto-segment", action="store_true", help="Compute metrics when conditioning requires them")
        cmd.add_argument("--segmentor-device", choices=["cpu", "cuda"], default="cpu")
        cmd.add_argument("--max-tiles", type=int, default=12)
        cmd.add_argument("--max-new-tokens", type=int)
        cmd.add_argument("--option-order", nargs=3, choices=["healthy", "stressed", "degraded"],
                         help="A/B/C mapping, only for an explicitly supplied reef-state MCQ prompt")
        cmd.add_argument("--output", type=Path, help="Output file; default: a unique run folder under results/<command>/")
        if name == "vqa":
            image_arguments(cmd)
            cmd.add_argument("--question", required=True)
            cmd.add_argument("--metrics", type=Path)
        elif name == "coralvqa":
            cmd.add_argument("--coralvqa-root", type=Path, default=CORALVQA_ROOT)
            cmd.add_argument("--annotations", type=Path, help="Optional equivalent CSV/JSONL export; frozen val_index.json is the default")
            cmd.add_argument("--start", type=int, default=0, help="Zero-based position in the frozen evaluation index")
            cmd.add_argument("--limit", type=int, help="Process only this many questions for a smoke test")
            cmd.add_argument("--metrics-dir", type=Path, help="Predicted measurements as <image-filename>.json")
            cmd.add_argument("--verify-image-hashes", action="store_true")
        else:
            cmd.add_argument("--input", type=Path, required=True, help="CSV with image,question and optional question_id,metrics")
            source = cmd.add_mutually_exclusive_group(required=True)
            source.add_argument("--images-root", type=Path)
            source.add_argument("--coralscapes-root", type=Path, nargs="?", const=CORALSCAPES_ROOT, help="CSV then requires split,index,question")
    for name in ("coralscapes-info", "coralscapes-export"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--coralscapes-root", type=Path, default=CORALSCAPES_ROOT)
        cmd.add_argument("--split", choices=["train", "validation", "test"], required=True)
        if name == "coralscapes-export":
            cmd.add_argument("--index", type=int, required=True)
            cmd.add_argument("--include-reference-mask", action="store_true")
            cmd.add_argument("--output", type=Path, help="Export directory; default: a unique run folder under results/coralscapes-export/")
    info = commands.add_parser("coralvqa-info", help="Validate the frozen evaluation index and image availability")
    info.add_argument("--coralvqa-root", type=Path, default=CORALVQA_ROOT)
    info.add_argument("--annotations", type=Path)
    info.add_argument("--verify-image-hashes", action="store_true")
    return p


def main(argv=None):
    from .pipeline import Pipeline, load_metrics
    args = parser().parse_args(argv)
    args.default_output = hasattr(args, "output") and args.output is None
    if args.default_output:
        directory = new_run_directory(RESULTS_ROOT, args.command, getattr(args, "model", None))
        if args.command in ("segment", "coralscapes-export"):
            args.output = directory
        else:
            args.output = directory / ("vqa.json" if args.command == "vqa" else "predictions.jsonl")
        print(f"Output: {args.output}", flush=True)
    if args.command == "coralvqa-info":
        from .coralvqa import CoralVQA
        dataset = CoralVQA(args.coralvqa_root, args.annotations)
        checked = dataset.validate_images(verify_hashes=args.verify_image_hashes)
        print(json.dumps({**dataset.describe(), "checked_images": checked,
                          "image_hashes_verified": args.verify_image_hashes}, indent=2))
        return
    if args.command.startswith("coralscapes-"):
        from .coralscapes import CoralScapes
        dataset = CoralScapes(args.coralscapes_root, args.split)
        if args.command == "coralscapes-info":
            print(json.dumps(dataset.describe(), indent=2))
        else:
            import numpy as np
            sample, label = dataset.get(args.index, args.include_reference_mask)
            # A new directory prevents accidental overwrites or mixing references with predictions.
            args.output.mkdir(parents=True, exist_ok=args.default_output)
            sample.rgb().save(args.output / "image.png")
            if label is not None:
                np.save(args.output / "reference_mask.npy", label, allow_pickle=False)
            metadata = {**sample.provenance, "encoded_image_sha256": sample.sha256,
                        "reference_mask_exported": label is not None}
            (args.output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Saved selected sample: {args.output}")
        return
    # Set before importing Transformers/Hugging Face through the model loaders.
    os.environ["HF_HUB_OFFLINE"] = "1" if args.offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if args.offline else "0"
    if args.command == "segment":
        pipeline = Pipeline(args.models_root, segmentor_device=args.device,
                            cache_dir=args.cache_dir, offline=args.offline)
        result = pipeline.measure(selected_image(args), output_dir=args.output, mask_path=args.mask)
        print(json.dumps(result, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    if args.command == "batch" and not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(args.models_root, args.model, args.adapter, args.device, args.precision,
                        args.quantization, args.segmentor_device, args.max_tiles, args.max_new_tokens,
                        args.cache_dir, args.offline)
    modes = list(dict.fromkeys(args.modes))
    if args.command == "coralvqa":
        from .coralvqa import CoralVQA
        from .conditioning import normalize_question
        dataset = CoralVQA(args.coralvqa_root, args.annotations)
        selected = dataset.select(args.start, args.limit)
        dataset.validate_images(selected, args.verify_image_hashes)
        partial = args.output.with_name(args.output.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"Previous partial output retained: {partial}")
        needs_metrics = any(mode != "none" for mode in modes)
        if needs_metrics and not args.auto_segment and args.metrics_dir is None:
            raise ValueError("Conditioned CoralVQA inference requires --auto-segment or --metrics-dir")
        with partial.open("x", encoding="utf-8") as output:
            for index, row in enumerate(selected, start=args.start):
                image = dataset.image_path(row)
                metrics = None
                if needs_metrics and args.metrics_dir is not None:
                    metrics = load_metrics(args.metrics_dir / (row["image"] + ".json"))
                for mode in modes:
                    result = pipeline.answer(image, normalize_question(row["question"]), mode,
                                             metrics, args.auto_segment, args.option_order)
                    result.update(dataset.metadata(row, index))
                    output.write(json.dumps(result) + "\n")
                    output.flush()
                    print(f"Saved {row['eval_row_id']}: {mode}", flush=True)
        partial.replace(args.output)
    elif args.command == "vqa":
        image = selected_image(args)
        metrics = load_metrics(args.metrics) if args.metrics else None
        results = [pipeline.answer(image, args.question, mode, metrics, args.auto_segment, args.option_order)
                   for mode in modes]
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
    else:
        partial = args.output.with_name(args.output.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"Previous partial output retained at {partial}; choose a new output path")
        with args.input.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            required = {"split", "index", "question"} if args.coralscapes_root else {"image", "question"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Batch CSV requires columns: {sorted(required)}")
            datasets = {}
            previous_key, previous_image = None, None
            count = 0
            with partial.open("x", encoding="utf-8") as output:
                for index, row in enumerate(reader):
                    if any(not row.get(field) for field in required):
                        raise ValueError(f"Empty image or question in CSV row {index + 2}")
                    if args.coralscapes_root:
                        from .coralscapes import CoralScapes
                        split = row["split"]
                        if split not in datasets:
                            datasets[split] = CoralScapes(args.coralscapes_root, split)
                        key = (split, int(row["index"]))
                        if key != previous_key:
                            previous_image = datasets[split].get(key[1])[0]
                            previous_key = key
                        image = previous_image
                    else:
                        image = args.images_root / row["image"]
                    metrics = load_metrics(args.input.parent / row["metrics"]) if row.get("metrics") else None
                    for mode in modes:
                        result = pipeline.answer(image, row["question"], mode, metrics, args.auto_segment, args.option_order)
                        result["question_id"] = row.get("question_id") or row.get("eval_row_id") or str(index)
                        if "answer" in row:
                            result["reference_answer"] = row["answer"]
                        for field in ("basic_visual_label", "ecological_label"):
                            if field in row:
                                result[field] = row[field]
                        result["input_row"] = index + 2
                        output.write(json.dumps(result) + "\n")
                        output.flush()
                        count += 1
                        print(f"Saved prediction {count}: row {index + 2}, {mode}", flush=True)
            if not count:
                partial.unlink()
                raise ValueError("Batch CSV contains no questions")
        partial.replace(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
