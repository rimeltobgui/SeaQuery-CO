"""Load the selected pinned base model on demand and attach local LoRA weights."""
from dataclasses import dataclass, fields
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

from ._common import checkpoint_files
from .paths import require_local_file

MODEL_IDS = {
    "segformer-b2": "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "internvl": "OpenGVLab/InternVL2_5-8B",
    "paligemma-pt": "google/paligemma2-3b-pt-448",
    "paligemma-mix": "google/paligemma2-3b-mix-448",
    "blip2": "Salesforce/blip2-flan-t5-xl",
}
MODEL_REVISIONS = {
    "segformer-b2": "9e150c56f013b1f07615a1f980b95f499b1bdc50",
    "llava": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "internvl": "e9e4c0dc1db56bfab10458671519b7fa3dd29463",
    "paligemma-pt": "bd60523c1ff2b0020c11e68affe65ef0b2379cab",
    "paligemma-mix": "1406c92ec87d32cc6b983239278901b904ba7a51",
    "blip2": "0eb0d3b46c14c1f8c7680bca2693baafdb90bb28",
}
VLM_NAMES = tuple(name for name in MODEL_IDS if name != "segformer-b2")
ADAPTERS = {"llava-lora": "llava", "llava-conditioned-lora": "llava",
            "internvl-lora": "internvl", "paligemma-mix-lora": "paligemma-mix"}


def default_cache_dir():
    root = Path("/scratch") if Path("/scratch").is_dir() else Path(tempfile.gettempdir())
    return root / "seaquery-model-cache"


def resolve_checkpoint(name, models_root, cache_dir=None, offline=False, save_models_to_data=False):
    """Prefer an existing complete local snapshot; otherwise use a pinned Hub ID."""
    if save_models_to_data:
        from .model_storage import require_code_ocean
        require_code_ocean(models_root)
    path = Path(models_root) / name
    if (path / "config.json").is_file():
        path = checkpoint_files(path)
        manifest = path / "seaquery_source.json"
        source = json.loads(manifest.read_text()) if manifest.is_file() else None
        return str(path), {"local_files_only": True}, {"location": "local", "manifest": source}
    if save_models_to_data:
        from .model_storage import saved_snapshot
        path = saved_snapshot(name, MODEL_IDS[name], MODEL_REVISIONS[name], models_root, offline)
        source = json.loads((path / "seaquery_source.json").read_text())
        return str(path), {"local_files_only": True}, {"location": "local", "manifest": source}
    cache = Path(cache_dir or default_cache_dir()).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    options = {"revision": MODEL_REVISIONS[name], "cache_dir": str(cache), "local_files_only": offline}
    source = {"location": "hub_cache", "repo_id": MODEL_IDS[name],
              "revision": MODEL_REVISIONS[name], "cache_dir": str(cache)}
    return MODEL_IDS[name], options, source


@dataclass
class LoadedModel:
    name: str
    model: object
    processor: object
    device: object
    dtype: object
    metadata: dict


def compatible_config(raw, supported):
    inactive = {"alora_invocation_tokens": None, "arrow_config": None,
                "corda_config": None, "ensure_weight_tying": False,
                "eva_config": None, "exclude_modules": None, "lora_bias": False,
                "lora_ga_config": None, "target_parameters": None,
                "trainable_token_indices": None, "use_bdlora": None, "use_qalora": False}
    clean = {}
    for key, value in raw.items():
        if key in supported:
            clean[key] = value
        elif key == "peft_version":
            continue
        elif key == "qalora_group_size" and not raw.get("use_qalora") and value == 16:
            continue
        elif key in inactive and value == inactive[key]:
            continue
        else:
            raise ValueError(f"Installed PEFT does not support active adapter field {key!r}")
    return clean


def adapter_config(path, expected_base):
    path = Path(path)
    raw = json.loads((path / "adapter_config.json").read_text())
    if raw.get("peft_type") != "LORA" or raw.get("base_model_name_or_path") != expected_base:
        raise ValueError(f"Adapter does not declare the selected base model: {expected_base}")
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(path / "adapter_model.safetensors")
    require_local_file(path / "adapter_model.safetensors")
    return raw


def resolve_runtime(name, device="auto", precision="auto"):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if precision == "auto":
        precision = "float32" if device == "cpu" or name == "blip2" else "float16"
    if device == "cpu" and precision != "float32":
        raise ValueError("CPU inference requires float32 in this implementation")
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Native BF16 is unavailable on this GPU; select float16 or float32")
    return torch.device(device), getattr(torch, precision)


def load_vlm(name, models_root, adapter=None, device="auto", precision="auto", quantization="none",
             cache_dir=None, offline=False, save_models_to_data=False):
    import torch
    from transformers import (AutoModel, AutoTokenizer, BitsAndBytesConfig,
                              LlavaForConditionalGeneration, LlavaProcessor,
                              PaliGemmaForConditionalGeneration, PaliGemmaProcessor,
                              Blip2ForConditionalGeneration, Blip2Processor)
    if version("transformers") != "4.46.3":
        raise RuntimeError("Use transformers==4.46.3 for these inference implementations")
    if name not in VLM_NAMES:
        raise ValueError(f"Unsupported VLM: {name}")
    path, source_options, source = resolve_checkpoint(name, models_root, cache_dir, offline, save_models_to_data)
    device, dtype = resolve_runtime(name, device, precision)
    adapter = Path(adapter).resolve() if adapter else None
    raw_adapter = adapter_config(adapter, MODEL_IDS[name]) if adapter else None
    if adapter and name == "blip2":
        raise ValueError("BLIP-2 adapters are not supported")
    if quantization not in ("none", "4bit"):
        raise ValueError("quantization must be none or 4bit")
    options = dict(**source_options, torch_dtype=dtype, low_cpu_mem_usage=True)
    options["device_map"] = {"": 0 if device.type == "cuda" else "cpu"}
    if quantization == "4bit":
        if device.type != "cuda":
            raise ValueError("4-bit loading requires CUDA")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Install bitsandbytes==0.45.5 to use --quantization 4bit") from error
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["vision_tower", "vision_model", "multi_modal_projector",
                                   "mlp1", "qformer", "language_projection", "lm_head"])
    if name == "internvl":
        model = AutoModel.from_pretrained(str(path), trust_remote_code=True,
                                         use_flash_attn=False, **options).eval()
        local_processor = adapter and (adapter / "tokenizer_config.json").is_file()
        proc_path = adapter if local_processor else path
        processor = AutoTokenizer.from_pretrained(str(proc_path),
                                                 **({"local_files_only": True} if local_processor else source_options),
                                                 trust_remote_code=True, use_fast=False)
    else:
        model_type, proc_type = {
            "llava": (LlavaForConditionalGeneration, LlavaProcessor),
            "paligemma-pt": (PaliGemmaForConditionalGeneration, PaliGemmaProcessor),
            "paligemma-mix": (PaliGemmaForConditionalGeneration, PaliGemmaProcessor),
            "blip2": (Blip2ForConditionalGeneration, Blip2Processor),
        }[name]
        model = model_type.from_pretrained(str(path), **options).eval()
        local_processor = adapter and (adapter / "preprocessor_config.json").is_file()
        proc_path = adapter if local_processor else path
        processor = proc_type.from_pretrained(str(proc_path),
                                             **({"local_files_only": True} if local_processor else source_options))
    if adapter:
        from peft import LoraConfig, PeftModel
        config = LoraConfig(**compatible_config(raw_adapter, {f.name for f in fields(LoraConfig)}))
        model = PeftModel.from_pretrained(model, str(adapter), config=config,
                                          is_trainable=False, local_files_only=True).eval()
        # Injected LoRA modules remain active; unwrapping exposes model-specific methods.
        model = model.get_base_model()
    if name == "internvl":
        token = processor.convert_tokens_to_ids("<IMG_CONTEXT>")
        if token is None or token == processor.unk_token_id:
            raise ValueError("InternVL tokenizer lacks <IMG_CONTEXT>")
        model.img_context_token_id = token
    if name == "llava":
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    metadata = {"model": name, "path": str(path), "adapter": str(adapter) if adapter else None,
                "device": str(device), "precision": str(dtype).removeprefix("torch."),
                "quantization": quantization, "source": source,
                "torch": torch.__version__, "transformers": version("transformers"),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    return LoadedModel(name, model, processor, device, dtype, metadata)
