from huggingface_hub import snapshot_download
from pathlib import Path
import os
import subprocess

def git_push_folder(folder_path, message):
    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "user.email", "github-actions[bot]@://github.com"], check=True)
        subprocess.run(["git", "add", folder_path], check=True)
        subprocess.run(["git", "commit", "-m", message], check=False)
        subprocess.run(["git", "push"], check=True)
        print(f"Successfully pushed {folder_path} to GitHub.")
    except Exception as e:
        print(f"Git push failed for {folder_path}: {e}")

p_cs = "data/datasets/coralscapes"
p_cvqa = "data/datasets/coralvqa"

Path(p_cs).mkdir(parents=True, exist_ok=True)
Path(p_cvqa).mkdir(parents=True, exist_ok=True)

if os.path.isdir(p_cs):
    try:
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
            max_workers=8,
        )
        print("Coralscapes download complete.")
        git_push_folder(p_cs, "Add coralscapes dataset")
    except Exception as e:
        print(f"Coralscapes download failed: {e}")

if os.path.isdir(p_cvqa):
    try:
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
            max_workers=8,
        )
        print("CoralVQA download complete.")
        git_push_folder(p_cvqa, "Add CoralVQA dataset")
    except Exception as e:
        print(f"CoralVQA download failed: {e}")
