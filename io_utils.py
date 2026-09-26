"""Atomic artifact writes and integrity checks for frozen search/training outputs."""
import csv
import hashlib
import json
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f'Artifact identity changed; use a new output directory: {path}')
    else:
        save_json(path, value)


def verify_files(folder, hashes):
    for name, expected in hashes.items():
        path = Path(folder) / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Frozen artifact is missing or changed: {path}')


def write_csv(path, rows):
    if not rows:
        raise ValueError('cannot write an empty result table')
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
