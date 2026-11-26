"""Local SegFormer-B2: float32, resize 1024x2048, two 1024-square crops."""
import json
from pathlib import Path
import numpy as np
from ._common import checkpoint_files, open_rgb
from .loaders import resolve_checkpoint

PROTOCOL = {"model": "segformer-b2", "precision": "float32", "resize_hw": [1024, 2048],
            "crop": 1024, "stride": 1024, "processor_do_resize": False,
            "denominator": "predicted pixels excluding labels 0 and 13",
            "cots_connectivity": 4, "cots_component_area": ">100 pixels"}


class Segmentor:
    def __init__(self, models_root, device="cpu", cache_dir=None, offline=False):
        import torch
        from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        path, source_options, self.source = resolve_checkpoint("segformer-b2", models_root, cache_dir, offline)
        if self.source["location"] == "local":
            config_path = Path(path) / "config.json"
        else:
            from huggingface_hub import hf_hub_download
            config_path = Path(hf_hub_download(path, "config.json", **source_options))
        raw = json.loads(config_path.read_text())
        # Repair label names only. Preserve the original 40 class indices.
        raw["id2label"] = {i: f"class_{i}" for i in range(40)}
        raw["label2id"] = {label: i for i, label in raw["id2label"].items()}
        config = SegformerConfig.from_dict(raw)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            str(path), config=config, torch_dtype=torch.float32, **source_options).to(self.device).eval()
        self.processor = SegformerImageProcessor.from_pretrained(str(path), **source_options)

    def predict(self, image):
        import torch
        from PIL import Image
        image = open_rgb(image).resize((2048, 1024), Image.Resampling.BILINEAR)
        pixels = self.processor(images=image, return_tensors="pt", do_resize=False)["pixel_values"]
        if pixels.shape[-2:] != (1024, 2048):
            raise ValueError("Segmentation input must be 1024x2048")
        pixels = pixels.to(self.device, torch.float32)
        pieces = []
        with torch.inference_mode():
            for left in (0, 1024):
                logits = self.model(pixel_values=pixels[:, :, :, left:left + 1024]).logits
                scores = torch.nn.functional.interpolate(logits.float(), size=(1024, 1024),
                                                         mode="bilinear", align_corners=False)
                pieces.append(scores.argmax(1)[0].cpu().numpy().astype(np.uint8))
        return np.concatenate(pieces, axis=1)
