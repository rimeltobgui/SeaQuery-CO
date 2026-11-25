from huggingface_hub import snapshot_download
from pathlib import Path
import os

p_cs="/data/coralscapes"
p_cvqa="/data/coralvqa"

Path(p_cs).mkdir(exist_ok=True)
Path(p_cvqa).mkdir(exist_ok=True)

if os.path.isdir(p_cs):
    snapshot_download(
        repo_id="EPFL-ECEO/coralscapes",
        repo_type="dataset",
        revision="ce6fa7f56460e1a8f683e4d73b2858145cf70020",
        local_dir=p_cs,
        allow_patterns=[
            "data/train-*.parquet",
            "data/validation-*.parquet",
            "data/test-*.parquet",
            "id2label.json",
            "label2color.json",
            "README.md",
            "LICENSE*",
        ],
        max_workers=2,
    )

    print("Coralscapes download complete.")

if os.path.isdir(p_cvqa):
    snapshot_download(
        repo_id="CoralReefData/CoralVQA",
        repo_type="dataset",
        revision="3da50a4429e474f75e887d63ffa99e348b6677e7",
        local_dir=p_cvqa,
        allow_patterns=[
            "CoralVQA_Image.zip",
            "CoralVQA.json",
            "CoralVQA_train.jsonl",
            "CoralVQA_test.jsonl",
            "annotations_file.zip",
            "README.md",
            "LICENSE*",
        ],
        max_workers=2,
    )

    print("CoralVQA download complete.")