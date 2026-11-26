"""Copy the frozen CoralVQA evaluation subset into a normal Drive folder."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def build(source, destination, workers=4):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((destination / 'subset_manifest.json').read_text())
    index = destination / 'val_index.json'
    if sha256(index) != manifest['source_sha256']['val_index.json']:
        raise RuntimeError('Frozen evaluation index checksum mismatch.')
    rows = json.loads(index.read_text())
    names = sorted({r['image'] for r in rows})
    if len(rows) != manifest['questions'] or len(names) != manifest['images']:
        raise RuntimeError('Evaluation count mismatch.')
    if any(Path(n).name != n or n in ('.', '..') for n in names):
        raise RuntimeError('Unsafe or unexpected image filenames.')
    images = destination / 'images'
    if source.resolve() == images.resolve():
        raise RuntimeError('Source and destination must differ.')
    expected = {r['image']: r for r in json.loads((destination / 'source_images.json').read_text())}
    if set(expected) != set(names):
        raise RuntimeError('Image manifest differs from the evaluation index.')
    missing = [n for n in names if not (source / n).is_file()]
    wrong_size = [n for n in names if (source / n).is_file() and (source / n).stat().st_size != expected[n]['bytes']]
    if missing or wrong_size:
        raise RuntimeError(f'Source validation failed: {len(missing)} missing, {len(wrong_size)} wrong size. Examples: {(missing + wrong_size)[:10]}')
    images.mkdir(parents=True, exist_ok=True)
    extras = {p.name for p in images.iterdir() if p.is_file() and not p.name.endswith('.partial')} - set(names)
    if extras:
        raise RuntimeError(f'Unexpected destination files; inspect before proceeding: {sorted(extras)[:10]}')
    completion = destination / 'COPY_COMPLETE.json'
    if completion.exists():
        completion.unlink()

    def copy_one(name):
        original, target = source / name, images / name
        source_hash = sha256(original)
        if not target.is_file() or sha256(target) != source_hash:
            temp = target.with_name(target.name + '.partial')
            shutil.copyfile(original, temp)
            if sha256(temp) != source_hash:
                raise RuntimeError(f'Copied image checksum mismatch: {name}')
            os.replace(temp, target)
        return {**expected[name], 'sha256': source_hash}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(copy_one, n): n for n in names}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 50 == 0 or len(results) == len(names):
                print(f'Verified {len(results)}/{len(names)} images', flush=True)
    results.sort(key=lambda r: r['image'])
    with (destination / 'image_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'bytes', 'source_drive_id', 'sha256'])
        writer.writeheader()
        writer.writerows(results)
    write_json(completion, {'status': 'complete', 'questions': len(rows), 'images': len(results), 'image_bytes': sum(r['bytes'] for r in results), 'all_copies_sha256_verified': True})
    print(f'Complete: {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/content/drive/MyDrive/PhD/CoralVQA/images'))
    parser.add_argument('--destination', type=Path, default=Path('/content/drive/MyDrive/PhD/CoralVQA/SeaQuery_CoralVQA_Evaluation'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    build(args.source, args.destination, args.workers)
