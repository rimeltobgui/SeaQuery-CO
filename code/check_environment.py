import json
import platform
from pathlib import Path
from importlib.metadata import version
import torch

report = {
    "python": platform.python_version(),
    "packages": {
        name: version(name)
        for name in [
            "torch", "torchvision", "transformers",
            "peft", "accelerate", "numpy", "scipy"
        ]
    },
    "cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    report.update(
        gpu=gpu.name,
        vram_gb=round(gpu.total_memory / 1024**3, 1),
        bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False),
    )
    x = torch.ones((32, 32), device="cuda")
    report["gpu_test_passed"] = (x @ x).sum().item() == 32768

destination = Path("/results")
destination.mkdir(parents=True, exist_ok=True)
(destination / "environment_check.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))

assert report["cuda_available"], "CUDA GPU is unavailable."
assert report["gpu_test_passed"], "GPU computation failed."