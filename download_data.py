import os
import sys
import time
import requests

DATA_DIR = "dataset"
ZIP_PATH = os.path.join(DATA_DIR, "captures1_v2.zip")
URL = "https://github.com/tjcruz-dei/ICS_PCAPS/releases/download/MODBUSTCP%231/captures1_v2.zip"

os.makedirs(DATA_DIR, exist_ok=True)

if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) == 669680240:
    print(f"File already downloaded ({ZIP_PATH}).")
    sys.exit(0)

print(f"Starting download from {URL} to {ZIP_PATH}...")
start_time = time.time()

with requests.get(URL, stream=True, allow_redirects=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))
    downloaded = 0
    last_print = time.time()
    
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 5:
                    pct = (downloaded / total_size * 100) if total_size else 0
                    speed = downloaded / (now - start_time) / (1024 * 1024)
                    print(f"Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                    last_print = now

elapsed = time.time() - start_time
print(f"Download complete: {downloaded / (1024*1024):.1f} MB in {elapsed:.1f}s ({downloaded / (1024*1024) / elapsed:.2f} MB/s)")
