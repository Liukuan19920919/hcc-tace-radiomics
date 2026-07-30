#!/usr/bin/env python3
"""Download files from Zenodo records via API."""
import json, os, sys, subprocess, argparse
from pathlib import Path

PROXY = os.environ.get('http_proxy', 'http://127.0.0.1:7897')

def download_record(record_id: str, output_dir: str, dry_run: bool = False):
    """Download all files from a Zenodo record."""
    import urllib.request

    api_url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(api_url)
    req.set_proxy(PROXY, 'http')

    print(f"Fetching {api_url} ...")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    title = data.get('title', 'Unknown')
    files = data.get('files', [])
    total = sum(f.get('size', 0) for f in files)

    print(f"Title: {title}")
    print(f"Files: {len(files)}, Total: {total//1024//1024}MB")
    print()

    os.makedirs(output_dir, exist_ok=True)

    if dry_run:
        for f in files:
            print(f"  {f['key']} ({f.get('size',0)//1024//1024}MB)")
        return

    success = 0
    for i, f in enumerate(files, 1):
        fname = f['key']
        url = f['links']['self']
        size = f.get('size', 0)
        outpath = os.path.join(output_dir, fname)

        if os.path.exists(outpath) and abs(os.path.getsize(outpath) - size) < 1000:
            print(f"  [{i}/{len(files)}] {fname} ✓ exists ({size//1024//1024}MB)")
            success += 1
            continue

        print(f"  [{i}/{len(files)}] {fname} ({size//1024//1024}MB) ...", end=" ", flush=True)

        cmd = [
            'curl', '-sL', '--connect-timeout', '30', '--max-time', '7200',
            '--retry', '3', '-x', PROXY,
            '-o', outpath, url
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=7300)

        if ret.returncode == 0 and os.path.exists(outpath):
            actual = os.path.getsize(outpath)
            print(f"✓ {actual//1024//1024}MB")
            success += 1
        else:
            print(f"✗ failed")

    print(f"\nDownloaded {success}/{len(files)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('record_id', help='Zenodo record ID')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--dry-run', '-n', action='store_true')
    args = parser.parse_args()

    download_record(args.record_id, args.output, args.dry_run)
