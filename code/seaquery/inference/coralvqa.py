"""Frozen SeaQuery CoralVQA evaluation subset, preserving row IDs and references."""
import csv
import hashlib
import json
from pathlib import Path

from .conditioning import normalize_question
from .paths import CORALVQA_ROOT, require_local_file


def sha256(path):
    with require_local_file(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_records(path):
    path = require_local_file(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix == ".json":
            records = json.load(stream)
        elif path.suffix == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix == ".csv":
            records = list(csv.DictReader(stream))
        else:
            raise ValueError("Annotations must be .json, .jsonl, or .csv")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty list of evaluation records")
    return records


def image_filename(value):
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"Expected an original image filename: {value!r}")
    return value


class CoralVQA:
    def __init__(self, root=CORALVQA_ROOT, annotations=None):
        self.root = Path(root).resolve()
        self.index = self.root / "val_index.json"
        self.index_sha256 = sha256(self.index)
        manifest_path = self.root / "subset_manifest.json"
        self.manifest = json.loads(require_local_file(manifest_path).read_text(encoding="utf-8"))
        if self.manifest.get("scope") != "evaluation_only":
            raise ValueError("Expected the SeaQuery evaluation-only CoralVQA subset")
        expected = self.manifest["source_sha256"]["val_index.json"]
        if self.index_sha256 != expected:
            raise ValueError("Frozen val_index.json checksum differs from subset_manifest.json")
        self.records = read_records(self.index)
        seen = set()
        for row in self.records:
            if not isinstance(row, dict):
                raise ValueError("Each evaluation record must be an object")
            row_id = row.get("eval_row_id")
            if not isinstance(row_id, str) or not row_id or row_id in seen:
                raise ValueError("Missing or duplicate frozen eval_row_id")
            seen.add(row_id)
            image_filename(row.get("image"))
            if not isinstance(row.get("question"), str) or not normalize_question(row["question"]):
                raise ValueError(f"Empty question: {row_id}")
            if not isinstance(row.get("answer"), str):
                raise ValueError(f"Missing reference answer: {row_id}")
        self.image_names = {row["image"] for row in self.records}
        if len(self.records) != self.manifest["questions"] or len(self.image_names) != self.manifest["images"]:
            raise ValueError("Evaluation counts differ from subset_manifest.json")
        # Training names here are provenance only; their images need not be downloaded.
        split_path = self.root / "coralvqa_split_imagegrouped.json"
        split_hash = self.manifest["source_sha256"].get(split_path.name)
        if split_hash and sha256(split_path) != split_hash:
            raise ValueError("Original image split checksum mismatch")
        split = json.loads(require_local_file(split_path).read_text(encoding="utf-8"))
        if not self.image_names <= set(split["val_images"]) or self.image_names & set(split["train_images"]):
            raise ValueError("Evaluation images violate the frozen image-grouped split")
        self.annotations = Path(annotations).resolve() if annotations else self.index
        # Alternative exports must represent the SAME ordered cohort; no accidental
        # re-splitting, deduplication, sorting, or answer normalization.
        if self.annotations != self.index:
            other = read_records(self.annotations)
            if len(other) != len(self.records):
                raise ValueError("Annotation export differs from the frozen cohort size")
            for index, (original, alternate) in enumerate(zip(self.records, other)):
                uid = alternate.get("eval_row_id", alternate.get("question_id"))
                fields = ("image", "question", "answer", "basic_visual_label", "ecological_label")
                if uid != original["eval_row_id"] or any(alternate.get(k) != original.get(k) for k in fields):
                    raise ValueError(f"Annotation export disagrees with frozen row {index}")
        self.annotations_sha256 = sha256(self.annotations)

    def __len__(self):
        return len(self.records)

    def select(self, start=0, limit=None):
        if start < 0 or start >= len(self) or (limit is not None and limit < 1):
            raise ValueError("start must select an existing row; limit must be positive")
        return self.records[start:start + limit if limit is not None else None]

    def image_path(self, row):
        path = self.root / "images" / image_filename(row["image"])
        path = require_local_file(path)
        if not path.resolve().is_relative_to((self.root / "images").resolve()):
            raise ValueError("Image resolves outside the subset images directory")
        return path

    def validate_images(self, records=None, verify_hashes=False):
        selected = self.records if records is None else records
        names = sorted({row["image"] for row in selected})
        sizes = {}
        manifest_file = self.root / "image_manifest.csv"
        if manifest_file.is_file():
            with require_local_file(manifest_file).open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    name = image_filename(row["image"])
                    if name in sizes:
                        raise ValueError("Duplicate image manifest entry")
                    sizes[name] = row
            if set(sizes) != self.image_names:
                raise ValueError("Image manifest filenames differ from the frozen cohort")
        elif verify_hashes:
            raise FileNotFoundError(manifest_file)
        for name in names:
            path = self.image_path({"image": name})
            if sizes and path.stat().st_size != int(sizes[name]["bytes"]):
                raise ValueError(f"Image byte size differs from manifest: {name}")
            if verify_hashes:
                expected = sizes[name].get("sha256")
                if not expected or sha256(path) != expected:
                    raise ValueError(f"Image SHA-256 mismatch or absent checksum: {name}")
        return len(names)

    def describe(self):
        return {"root": str(self.root), "scope": "evaluation_only", "questions": len(self),
                "images": len(self.image_names), "annotations": str(self.annotations),
                "annotations_sha256": self.annotations_sha256, "frozen_index_sha256": self.index_sha256}

    def metadata(self, row, index):
        return {"question_id": row["eval_row_id"], "eval_row_id": row["eval_row_id"],
                "reference_answer": row["answer"], "question_raw": row["question"],
                "question_normalization": "remove <image> and [vqa]; collapse whitespace; preserve case",
                "basic_visual_label": row.get("basic_visual_label"),
                "ecological_label": row.get("ecological_label"),
                "pair_occurrence": row.get("pair_occurrence"), "input_row": index,
                "dataset": "SeaQuery CoralVQA evaluation subset",
                "dataset_source": self.describe()}
