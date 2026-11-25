import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODELS = {
    "segformer-b2": (
        "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024",
        "9e150c56f013b1f07615a1f980b95f499b1bdc50",
    ),
    "llava": (
        "llava-hf/llava-1.5-7b-hf",
        "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    ),
    "internvl": (
        "OpenGVLab/InternVL2_5-8B",
        "e9e4c0dc1db56bfab10458671519b7fa3dd29463",
    ),
    "paligemma-pt": (
        "google/paligemma2-3b-pt-448",
        "bd60523c1ff2b0020c11e68affe65ef0b2379cab",
    ),
    "paligemma-mix": (
        "google/paligemma2-3b-mix-448",
        "1406c92ec87d32cc6b983239278901b904ba7a51",
    ),
    "blip2": (
        "Salesforce/blip2-flan-t5-xl",
        "0eb0d3b46c14c1f8c7680bca2693baafdb90bb28",
    ),
}


def download_model(name, root, api):
    repo_id, revision = MODELS[name]
    destination = root / name

    info = api.model_info(repo_id, revision=revision)
    if info.sha != revision:
        raise RuntimeError("The returned revision does not match the requested one.")

    filenames = [item.rfilename for item in info.siblings]

    # Download one weight format, avoiding duplicate PyTorch weights.
    has_safetensors = any(
        "/" not in filename and filename.endswith(".safetensors")
        for filename in filenames
    )
    weight_pattern = "*.safetensors" if has_safetensors else "*.bin"

    print(f"\nDownloading {name} → {destination}", flush=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
        allow_patterns=[
            weight_pattern,
            "*.json",
            "*.model",
            "*.txt",
            "*.py",
            "*.md",
            "*.jinja",
            "*.tiktoken",
            "*.vocab",
            "*.merges",
            "LICENSE*",
            "NOTICE*",
        ],
        max_workers=2,
    )

    weights = list(destination.glob(weight_pattern))
    if not weights or not (destination / "config.json").is_file():
        raise RuntimeError("Download is missing model weights or config.json.")

    # Check that all shards referenced by the selected weight index exist.
    index_name = (
        "model.safetensors.index.json"
        if has_safetensors
        else "pytorch_model.bin.index.json"
    )
    index_path = destination / index_name

    if index_path.exists():
        index = json.loads(index_path.read_text())
        missing = [
            shard
            for shard in set(index["weight_map"].values())
            if not (destination / shard).is_file()
        ]
        if missing:
            raise RuntimeError(f"Missing weight shards: {missing}")

    manifest = {
        "name": name,
        "repo_id": repo_id,
        "revision": revision,
        "source": f"https://huggingface.co/{repo_id}",
        "weight_format": "safetensors" if has_safetensors else "pytorch_bin",
    }
    (destination / "seaquery_source.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"Completed: {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/models"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    failures = []

    for name in dict.fromkeys(args.models):
        try:
            download_model(name, args.root, api)
        except Exception as error:
            failures.append(name)
            print(f"\nFAILED: {name}: {error}", flush=True)

    if failures:
        raise SystemExit(
            "\nIncomplete downloads: " + ", ".join(failures)
            + "\nResolve the reported errors and rerun the command."
        )

    print("\nAll selected models downloaded successfully.")


if __name__ == "__main__":
    main()
