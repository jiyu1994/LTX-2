"""Convert diffsynth-studio Wan dataset metadata to LTX-2 trainer format.

diffsynth format:
  [{"prompt": "...", "image": "images/xxx.jpg", "video": "videos/xxx.mp4"}, ...]

LTX-2 format:
  [{"caption": "...", "media_path": "/abs/path/to/videos/xxx.mp4"}, ...]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def convert(
    input_path: Path,
    dataset_root: Path,
    output_path: Path,
) -> None:
    records: List[Dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))

    converted: List[Dict[str, str]] = []
    skipped = 0

    for item in records:
        prompt = item.get("prompt", "").strip()
        video_rel = item.get("video")

        if not prompt or not video_rel:
            skipped += 1
            continue

        video_abs = dataset_root / video_rel
        if not video_abs.exists():
            print(f"[WARN] video not found, skip: {video_abs}")
            skipped += 1
            continue

        converted.append(
            {
                "caption": prompt,
                "media_path": str(video_abs),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Total input:   {len(records)}")
    print(f"Converted:     {len(converted)}")
    print(f"Skipped:       {skipped}")
    print(f"Output:        {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert diffsynth-studio Wan dataset metadata to LTX-2 format."
    )
    parser.add_argument("input_json", type=Path, help="diffsynth metadata JSON (e.g. metadata_clean.json)")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory of the Wan dataset")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: <dataset-root>/dataset_ltx2.json)")
    args = parser.parse_args()

    output = args.output or (args.dataset_root / "dataset_ltx2.json")
    convert(args.input_json, args.dataset_root, output)


if __name__ == "__main__":
    main()
