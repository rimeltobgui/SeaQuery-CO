"""Tensor and image helpers shared by the inference backends."""
from pathlib import Path
from .paths import require_local_file


def open_rgb(path):
    from PIL import Image
    from .coralscapes import ParquetImage
    if isinstance(path, ParquetImage):
        return path.rgb()
    with Image.open(require_local_file(path)) as image:
        return image.convert("RGB")


def move_inputs(inputs, device, dtype):
    return {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in inputs.items()}


def insert_metrics(embeds, attention, metric_embeds, position):
    import torch
    mask = torch.ones(metric_embeds.shape[:2], dtype=attention.dtype, device=attention.device)
    return (torch.cat([embeds[:, :position], metric_embeds, embeds[:, position:]], dim=1),
            torch.cat([attention[:, :position], mask, attention[:, position:]], dim=1))


def replace_image_embeddings(embeds, ids, token_id, features):
    import torch
    mask = ids.eq(token_id).unsqueeze(-1).expand_as(embeds)
    expected = features.numel() // embeds.shape[-1]
    actual = int(mask.sum().item()) // embeds.shape[-1]
    if actual != expected or not actual:
        raise ValueError(f"Image token/feature mismatch: {actual} tokens, {expected} features")
    return embeds.masked_scatter(mask, features.to(device=embeds.device, dtype=embeds.dtype))


def checkpoint_files(path):
    import json
    path = Path(path)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing local config.json: {path}")
    if not list(path.glob("*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"Missing local model weights: {path}")
    for weights in list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin")):
        require_local_file(weights)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if index.exists():
            for filename in set(json.loads(index.read_text())["weight_map"].values()):
                if not (path / filename).is_file():
                    raise FileNotFoundError(f"Missing weight shard: {path / filename}")
    return path.resolve()
