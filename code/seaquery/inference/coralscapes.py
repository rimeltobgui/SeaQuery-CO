"""Read local original CoralScapes Parquet shards without extracting the dataset."""
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from PIL import Image
from .paths import require_local_file


@dataclass(frozen=True)
class ParquetImage:
    encoded: bytes
    provenance: dict

    @property
    def sha256(self):
        return hashlib.sha256(self.encoded).hexdigest()

    @property
    def name(self):
        p = self.provenance
        return f"{p['shard']}#row={p['shard_row']}"

    def rgb(self):
        with Image.open(BytesIO(self.encoded)) as image:
            return image.convert("RGB")


def _bytes(value, column):
    # HF Image storage is a struct with bytes and path. Do not follow archived
    # absolute paths: these may identify the author's original machine.
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise ValueError(f"{column} has no embedded image bytes; expected original CoralScapes Parquet")
    return bytes(value)


class CoralScapes:
    """Split index is zero-based over numerically ordered shards, then rows.

    Construction reads footers only. get() streams batches of one row from the
    selected row group and projects only requested columns. Parquet may still
    decompress a column page; the complete split is never materialized.
    """
    def __init__(self, root, split="test"):
        import pyarrow.parquet as pq
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        self.root = Path(root).resolve()
        self.split = split
        self.directory = self.root / "data" if (self.root / "data").is_dir() else self.root
        pattern = re.compile(rf"{split}-(\d+)-of-(\d+)\.parquet")
        matches = []
        for path in self.directory.glob(f"{split}-*.parquet"):
            match = pattern.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unrecognized shard filename: {path.name}")
            matches.append((int(match[1]), int(match[2]), path))
        if not matches:
            raise FileNotFoundError(f"No {split}-*.parquet shards in {self.directory}")
        matches.sort()
        total = matches[0][1]
        if total < 1 or any(n != total for _, n, _ in matches) or [i for i, _, _ in matches] != list(range(total)):
            raise ValueError(f"Incomplete or duplicate {split} shards; retain all {total} shards for stable row indices")
        self.shards = []
        start = 0
        for _, _, path in matches:
            require_local_file(path)
            with pq.ParquetFile(path) as file:
                if not {"image", "label"}.issubset(file.schema_arrow.names):
                    raise ValueError(f"Expected image and label columns: {path}")
                groups = [file.metadata.row_group(i).num_rows for i in range(file.num_row_groups)]
                count = file.metadata.num_rows
            self.shards.append((path, start, count, groups))
            start += count
        self.count = start

    def __len__(self):
        return self.count

    def describe(self):
        return {"split": self.split, "rows": self.count, "directory": str(self.directory),
                "shards": [{"file": str(p), "rows": n, "first_index": s} for p, s, n, _ in self.shards]}

    def get(self, index, include_label=False):
        import pyarrow.parquet as pq
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(f"{self.split} index must be in [0, {self.count - 1}]")
        path, start, _, groups = next(s for s in self.shards if s[1] <= index < s[1] + s[2])
        shard_row = index - start
        offset = shard_row
        for group, count in enumerate(groups):
            if offset < count:
                break
            offset -= count
        columns = ["image", "label"] if include_label else ["image"]
        with pq.ParquetFile(path) as file:
            for i, batch in enumerate(file.iter_batches(batch_size=1, row_groups=[group], columns=columns, use_threads=False)):
                if i == offset:
                    row = batch.to_pylist()[0]
                    break
            else:
                raise RuntimeError(f"Missing Parquet row {shard_row}: {path}")
        cell = row["image"]
        original = cell.get("path") if isinstance(cell, dict) else None
        sample = ParquetImage(_bytes(cell, "image"), {
            "dataset": "EPFL-ECEO/coralscapes", "split": self.split,
            "split_index": index, "shard": str(path), "shard_row": shard_row,
            "row_group": group, "row_in_group": offset, "original_image_path": original,
        })
        image = sample.rgb()  # Validate decoding before model loading.
        label = None
        if include_label:
            with Image.open(BytesIO(_bytes(row["label"], "label"))) as mask:
                # Preserve palette indices; converting to RGB destroys class IDs.
                label = np.array(mask)
            if label.ndim != 2 or label.dtype.kind not in "ui" or label.size == 0 or label.min() < 0 or label.max() > 39:
                raise ValueError("Reference label must be a 2-D integer mask with class IDs 0..39")
            if label.shape != (image.height, image.width):
                raise ValueError("Reference label and image dimensions do not match")
        return sample, label
