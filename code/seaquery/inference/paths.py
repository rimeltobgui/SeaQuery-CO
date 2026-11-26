"""Default paths shared by the local repository and Code Ocean capsule."""
from pathlib import Path
from datetime import datetime, timezone
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
DATASETS_ROOT = DATA_ROOT / "datasets"
CORALSCAPES_ROOT = DATASETS_ROOT / "coralscapes"
CORALVQA_ROOT = DATASETS_ROOT / "coralvqa"
MODELS_ROOT = DATA_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"


def new_run_directory(results_root, command, model=None):
    """Atomically reserve a unique directory, including for concurrent runs."""
    parent = Path(results_root) / command
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = f"{model}-{stamp}-" if model else f"{stamp}-"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def require_local_file(path):
    """Fail clearly for missing assets and Git LFS pointer text."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        header = stream.read(100)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"Git LFS pointer, not downloaded data: {path}. "
            "In your existing local checkout run git lfs pull, or attach the actual file in Code Ocean.")
    return path
