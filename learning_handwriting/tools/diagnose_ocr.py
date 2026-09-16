"""Diagnose PaddleOCR input and inference independently of its CLI wrapper."""
from __future__ import annotations
import os
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "ocr_debug"

def package(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"

def main():
    print("=== Environment ===")
    print("python:", sys.executable)
    print("paddlepaddle:", package("paddlepaddle"))
    print("paddleocr:", package("paddleocr"))
    print("raw directory:", RAW, "exists=", RAW.is_dir())
    images = sorted([*RAW.glob("*.jpg"), *RAW.glob("*.jpeg"), *RAW.glob("*.png")])
    print("image count:", len(images))
    if not images:
        raise SystemExit("No JPG/JPEG/PNG image found in data/raw")
    image = images[0].resolve()
    print("test image:", image)
    print("exists=", image.is_file(), "bytes=", image.stat().st_size)

    from paddleocr import PaddleOCR
    print("=== Creating pipeline (MKLDNN disabled) ===")
    ocr = PaddleOCR(
        lang="korean",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print("=== Calling predict() ===")
    OUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for count, result in enumerate(ocr.predict(str(image)), start=1):
        print(f"result #{count}: {type(result).__name__}")
        print(result)
        if hasattr(result, "save_to_json"):
            result.save_to_json(str(OUT))
            print("saved JSON to:", OUT)
    print("=== Finished; result count:", count, "===")
    if count == 0:
        print("DIAGNOSIS: The API returned zero results. Copy this entire output.")

if __name__ == "__main__":
    main()
