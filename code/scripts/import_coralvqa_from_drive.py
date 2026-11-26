"""Import the verified SeaQuery evaluation folder using read-only Drive access."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time

API = 'https://www.googleapis.com/drive/v3/files'
FOLDER = 'application/vnd.google-apps.folder'
IMAGE_MANIFEST_SHA256 = '886de34c04f535803bbf6a7cec526c0a312dfbe209a4a968888dc254107abeeb'
EVAL_INDEX_SHA256 = 'd08b9ecb5fa4527bbda38c1ab6d481ed485d983b5f48e09c6208b6021adf34b4'
EXPECTED_IMAGES = 1037
EXPECTED_QUESTIONS = 5158
METADATA = (
    'COPY_COMPLETE.json', 'image_manifest.csv', 'subset_manifest.json',
    'source_images.json', 'val_index.json', 'coralvqa_split_imagegrouped.json',
    'coralvqa_eval.jsonl', 'coralvqa_eval.csv', 'README.md',
    'build_evaluation_subset.py', 'Build_CoralVQA_Evaluation_Subset.ipynb',
)


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def safe_name(name):
    if not name or name in {'.', '..'} or '/' in name or '\\' in name or '\0' in name:
        raise ValueError('Unexpected Drive filename')
    return name


def unique_files(files):
    result = {}
    for entry in files:
        name = safe_name(entry['name'])
        if name in result:
            raise ValueError(f'Duplicate Drive filename: {name}')
        result[name] = entry
    return result


class Drive:
    def __init__(self, account):
        self.account = account
        self.local = threading.local()

    def session(self):
        if not hasattr(self.local, 'session'):
            from google.auth.transport.requests import AuthorizedSession
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(
                self.account, scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            self.local.session = AuthorizedSession(credentials)
        return self.local.session

    def request(self, url, **kwargs):
        for attempt in range(6):
            response = self.session().get(url, timeout=(30, 180), **kwargs)
            retry = response.status_code in {429, 500, 502, 503, 504}
            if response.status_code == 403:
                retry = any(reason in response.text for reason in
                            ('rateLimitExceeded', 'userRateLimitExceeded'))
            if retry and attempt < 5:
                response.close()
                time.sleep(2 ** attempt)
                continue
            if not response.ok:
                status = response.status_code
                response.close()
                raise RuntimeError(f'Drive returned HTTP {status}. Check API access and folder sharing.')
            return response
        raise RuntimeError('Drive retry limit reached')

    def list_files(self, folder_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', folder_id):
            raise ValueError('Invalid folder ID')
        files, token = [], None
        while True:
            params = {'q': f"'{folder_id}' in parents and trashed = false",
                      'pageSize': 1000,
                      'fields': 'nextPageToken,incompleteSearch,files(id,name,mimeType,size,md5Checksum)',
                      'supportsAllDrives': 'true', 'includeItemsFromAllDrives': 'true'}
            if token:
                params['pageToken'] = token
            with self.request(API, params=params) as response:
                page = response.json()
            if page.get('incompleteSearch'):
                raise RuntimeError('Drive returned an incomplete folder listing')
            files.extend(page.get('files', []))
            token = page.get('nextPageToken')
            if not token:
                return files

    def download(self, entry, path):
        if entry['mimeType'] == FOLDER or 'size' not in entry:
            raise ValueError(f"Expected a downloadable file: {entry['name']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.partial')
        with self.request(f"{API}/{entry['id']}",
                          params={'alt': 'media', 'supportsAllDrives': 'true'}, stream=True) as response:
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(chunk_size=4 << 20):
                    stream.write(chunk)
        if temporary.stat().st_size != int(entry['size']):
            raise RuntimeError(f"Incomplete download: {entry['name']}")
        if entry.get('md5Checksum') and digest(temporary, 'md5') != entry['md5Checksum']:
            raise RuntimeError(f"Drive checksum mismatch: {entry['name']}")
        os.replace(temporary, path)


def verify_metadata(stage, expected_images=EXPECTED_IMAGES, expected_questions=EXPECTED_QUESTIONS,
                    manifest_hash=IMAGE_MANIFEST_SHA256, index_hash=EVAL_INDEX_SHA256):
    if digest(stage / 'image_manifest.csv') != manifest_hash:
        raise ValueError('Image manifest differs from the frozen study manifest')
    if digest(stage / 'val_index.json') != index_hash:
        raise ValueError('Evaluation index differs from the frozen study index')
    complete = json.loads((stage / 'COPY_COMPLETE.json').read_text())
    if (complete.get('status') != 'complete' or complete.get('images') != expected_images
            or complete.get('questions') != expected_questions
            or complete.get('all_copies_sha256_verified') is not True):
        raise ValueError('Drive subset is incomplete or has unexpected counts')
    with (stage / 'image_manifest.csv').open(newline='') as stream:
        records = list(csv.DictReader(stream))
    expected = {}
    for row in records:
        name = safe_name(row['image'])
        if name in expected or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid or duplicate image manifest entry')
        expected[name] = {'bytes': int(row['bytes']), 'sha256': row['sha256']}
        if expected[name]['bytes'] <= 0:
            raise ValueError('Invalid image byte size')
    rows = json.loads((stage / 'val_index.json').read_text())
    if len(rows) != expected_questions or len(expected) != expected_images:
        raise ValueError('Unexpected evaluation counts')
    if {r['image'] for r in rows} != set(expected):
        raise ValueError('Evaluation annotations and image manifest disagree')
    if sum(r['bytes'] for r in expected.values()) != complete['image_bytes']:
        raise ValueError('Total image byte count differs from completion record')
    with (stage / 'coralvqa_eval.jsonl').open() as stream:
        if [json.loads(line) for line in stream if line.strip()] != rows:
            raise ValueError('JSONL annotations differ from frozen evaluation index')
    with (stage / 'coralvqa_eval.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    fields = ('image', 'question', 'answer', 'basic_visual_label', 'ecological_label')
    if len(csv_rows) != len(rows) or any(
        a['question_id'] != b['eval_row_id'] or any(a[k] != b[k] for k in fields)
        for a, b in zip(csv_rows, rows)
    ):
        raise ValueError('CSV annotations differ from frozen evaluation index')
    return expected


def import_subset(drive, folder_id, destination, workers=4):
    root = unique_files(drive.list_files(folder_id))
    missing = set(METADATA) - root.keys()
    if missing or root.get('images', {}).get('mimeType') != FOLDER:
        raise ValueError(f'Missing subset files or images folder: {sorted(missing)}')
    destination = Path(destination)
    for parent in (destination, *destination.parents):
        if parent.is_symlink():
            raise ValueError('Destination must not contain symbolic links')
    if destination.exists() and any(p.is_symlink() for p in destination.rglob('*')):
        raise ValueError('Existing destination contains symbolic links')
    with tempfile.TemporaryDirectory(prefix='coralvqa-import-') as temporary:
        stage = Path(temporary)
        for name in METADATA:
            drive.download(root[name], stage / name)
        expected = verify_metadata(stage)
        images = unique_files(drive.list_files(root['images']['id']))
        if set(images) != set(expected):
            raise ValueError('Drive image folder has missing or additional files')
        if any(int(images[n].get('size', -1)) != r['bytes'] for n, r in expected.items()):
            raise ValueError('Drive image sizes disagree with the frozen manifest')

        def fetch_image(name):
            path = stage / 'images' / name
            drive.download(images[name], path)
            if digest(path) != expected[name]['sha256']:
                raise ValueError(f'Image SHA-256 mismatch: {name}')

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_image, name) for name in expected]
            for number, future in enumerate(as_completed(futures), 1):
                future.result()
                if number % 50 == 0 or number == len(expected):
                    print(f'Verified {number}/{len(expected)} images', flush=True)
        image_dir = destination / 'images'
        if image_dir.exists() and {p.name for p in image_dir.iterdir()} - set(expected):
            raise ValueError('Destination images folder contains unrelated files; nothing replaced')
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, destination, dirs_exist_ok=True)
        print(f'Imported {EXPECTED_IMAGES} images and {EXPECTED_QUESTIONS} questions into {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder-id', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    secret = os.environ.get('GDRIVE_SERVICE_ACCOUNT_JSON', '')
    if not secret:
        parser.error('GDRIVE_SERVICE_ACCOUNT_JSON is missing')
    try:
        account = json.loads(secret)
    except json.JSONDecodeError:
        parser.error('GDRIVE_SERVICE_ACCOUNT_JSON must contain a JSON service-account key')
    import_subset(Drive(account), args.folder_id, args.destination, args.workers)


if __name__ == '__main__':
    main()
