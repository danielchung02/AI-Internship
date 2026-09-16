"""Free local batch OCR fallback for Windows when PaddleOCR cannot infer.

Writes one JSON file per source page with word/line polygons, text, and confidence.
These are OCR *draft labels*, not final character labels for model training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "ocr_easyocr")
    ap.add_argument("--limit", type=int, default=0, help="Process only this many pages; 0 means all")
    ap.add_argument("--gpu", action="store_true", help="Use local CUDA Torch, if configured")
    args = ap.parse_args()
    try:
        import easyocr
    except ImportError:
        raise SystemExit("EasyOCR is missing. Run: python -m pip install easyocr")
    images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading Korean + English OCR model; pages={len(images)}, gpu={args.gpu}")
    reader = easyocr.Reader(["ko", "en"], gpu=args.gpu)
    for index, image in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {image.name}")
        raw = reader.readtext(str(image), detail=1, paragraph=False)
        results = [
            {"polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in box],
             "text": text, "confidence": round(float(confidence), 5)}
            for box, text, confidence in raw
        ]
        payload = {"source": str(image.resolve()), "engine": "easyocr", "languages": ["ko", "en"], "results": results}
        (args.output / f"{image.stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done. JSON drafts:", args.output)

if __name__ == "__main__":
    main()
