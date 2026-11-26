"""Opt-in model snapshots in Code Ocean's writable /data/models folders."""
import json
from pathlib import Path
import re


def require_code_ocean(models_root):
    code_root = Path(__file__).resolve().parents[2]
    target = Path(models_root).resolve()
    if code_root != Path('/code') or not target.is_relative_to(Path('/data')):
        raise ValueError('--code-ocean-save-models is only supported in the Code Ocean /code and /data layout')


def saved_snapshot(name, repo_id, revision, models_root, offline=False):
    from filelock import FileLock
    from huggingface_hub import list_repo_files, snapshot_download
    from ._common import checkpoint_files

    root = Path(models_root).resolve()
    require_code_ocean(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(root / f'.{name}.download.lock'))
        with lock:
            target = root / name
            if target.exists():
                checkpoint_files(target)
                return target
            if offline:
                raise FileNotFoundError(f'No saved snapshot at {target}; disable --offline for its first download')
            staging = root / f'.{name}.download'
            staging.mkdir(exist_ok=True)
            print(f'Saving selected model to {target}. Use a Code Ocean Cloud Workstation to retain /data changes.', flush=True)
            files = list_repo_files(repo_id, revision=revision, repo_type='model')
            safe = [f for f in files if re.fullmatch(r'model(?:-\d+-of-\d+)?\.safetensors', f)]
            weights = safe or [f for f in files if re.fullmatch(r'pytorch_model(?:-\d+-of-\d+)?\.bin', f)]
            if not weights:
                raise ValueError(f'No supported model weights in {repo_id} at {revision}')
            support = [f for f in files if f.endswith(('.json', '.model', '.txt', '.py', '.tiktoken')) and '/' not in f]
            # Keep only the weight index matching the chosen format.
            unused_index = 'pytorch_model.bin.index.json' if safe else 'model.safetensors.index.json'
            support = [f for f in support if f != unused_index]
            selected = sorted(set(weights + support))
            snapshot_download(repo_id, revision=revision, local_dir=str(staging),
                              allow_patterns=selected, max_workers=2)
            # Never expose a partly downloaded folder as a usable local checkpoint.
            for filename in selected:
                if not (staging / filename).is_file():
                    raise FileNotFoundError(staging / filename)
            checkpoint_files(staging)
            manifest = {'repo_id': repo_id, 'revision': revision, 'files': selected,
                        'storage': 'code_ocean_data', 'complete': True}
            (staging / 'seaquery_source.json').write_text(json.dumps(manifest, indent=2) + '\n')
            staging.rename(target)
            return target
    except (PermissionError, OSError) as error:
        if isinstance(error, FileNotFoundError):
            raise
        raise RuntimeError(f'Cannot save model under {root}: {error}. Use writable capsule data in a Cloud Workstation, not a read-only attached asset.') from error
