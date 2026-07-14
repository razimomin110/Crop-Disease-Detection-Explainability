"""Download the PlantVillage dataset (color images subset) from GitHub."""
import io
import os
import shutil
import zipfile

import requests

# The PlantVillage dataset on GitHub (color images)
GITHUB_ZIP_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"

TARGET_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Apple___Apple_scab",
    "Apple___healthy",
]

DEST_DIR = os.path.join(os.path.dirname(__file__), "PlantVillage")


def download_dataset():
    os.makedirs(DEST_DIR, exist_ok=True)

    print(f"[DOWNLOAD] Downloading PlantVillage dataset from GitHub...")
    print(f"[DOWNLOAD] URL: {GITHUB_ZIP_URL}")
    print(f"[DOWNLOAD] This may take several minutes (~800 MB)...")

    resp = requests.get(GITHUB_ZIP_URL, stream=True, timeout=600)
    resp.raise_for_status()

    # Read into memory (or save to temp file for very large downloads)
    total_size = int(resp.headers.get("content-length", 0))
    print(f"[DOWNLOAD] Total size: {total_size / 1024 / 1024:.1f} MB")

    chunks: list[bytes] = []
    downloaded: int = 0
    for chunk_raw in resp.iter_content(chunk_size=8192 * 16):
        if isinstance(chunk_raw, bytes):
            chunk: bytes = chunk_raw
            chunks.append(chunk)
            
            # pyre-ignore[58]
            downloaded += int(len(chunk))

            if total_size:
                pct = downloaded / total_size * 100
                print(f"\r[DOWNLOAD] {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="", flush=True)

    print("\n[DOWNLOAD] Download complete. Extracting...")

    data = b"".join(chunks)
    zf = zipfile.ZipFile(io.BytesIO(data))

    # The zip contains: PlantVillage-Dataset-master/raw/color/<class_name>/...
    color_prefix = "PlantVillage-Dataset-master/raw/color/"
    extracted = 0

    for info in zf.infolist():
        if not info.filename.startswith(color_prefix):
            continue
        # Get the relative path after color/
        rel = info.filename.removeprefix(color_prefix)
        if not rel or info.is_dir():
            continue

        # Check if it belongs to one of our target classes
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        class_name = parts[0]
        if class_name not in TARGET_CLASSES:
            continue

        # Extract
        dest_path = os.path.join(DEST_DIR, rel)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(zf.read(info.filename))
        # pyre-ignore[58]
        extracted += 1

    print(f"[DOWNLOAD] Extracted {extracted} images into {DEST_DIR}")

    # Print per-class counts
    for cls in sorted(TARGET_CLASSES):
        cls_dir = os.path.join(DEST_DIR, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"  {cls}: {count} images")
        else:
            print(f"  {cls}: NOT FOUND")


if __name__ == "__main__":
    download_dataset()
