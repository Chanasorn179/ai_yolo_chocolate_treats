import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset_sam"
DEFAULT_MODEL = "sam2.1_t.pt"
DEFAULT_PAD = 0.12
DEFAULT_MIN_IOU = 0.6
DEFAULT_MIN_AREA_RATIO = 0.6
DEFAULT_MAX_AREA_RATIO = 1.8
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tighten YOLO boxes with SAM: each box, padded slightly, prompts SAM "
            "and the mask's bounding box becomes the new label."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SAM weights (downloaded on first use)")
    parser.add_argument("--pad", type=float, default=DEFAULT_PAD,
                        help="grow each prompt box by this fraction of its size on every side")
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU,
                        help="keep the original box when the SAM box overlaps it less than this")
    parser.add_argument("--min-area-ratio", type=float, default=DEFAULT_MIN_AREA_RATIO)
    parser.add_argument("--max-area-ratio", type=float, default=DEFAULT_MAX_AREA_RATIO)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must differ from --input-dir")
    if args.pad < 0:
        parser.error("--pad must not be negative")
    if not 0.0 < args.min_area_ratio < 1.0 < args.max_area_ratio:
        parser.error("need --min-area-ratio < 1 < --max-area-ratio")
    return args


def read_labels(label_path: Path):
    """Read YOLO label lines as (class_id, x_center, y_center, width, height)."""
    if not label_path.is_file():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def to_xyxy(row, width, height, pad=0.0):
    _, x, y, w, h = row
    pw, ph = w * pad, h * pad
    return [
        max(0.0, (x - w / 2 - pw) * width),
        max(0.0, (y - h / 2 - ph) * height),
        min(width, (x + w / 2 + pw) * width),
        min(height, (y + h / 2 + ph) * height),
    ]


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def refine_image(model, image_path, rows, args):
    """Return refined rows and how many boxes were kept unchanged."""
    if not rows:
        return [], 0
    with Image.open(image_path) as image:
        width, height = image.size
    prompts = [to_xyxy(row, width, height, args.pad) for row in rows]
    result = model(str(image_path), bboxes=prompts, device=args.device, verbose=False)[0]
    masks = result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else []
    mask_h, mask_w = (masks[0].shape if len(masks) else (height, width))
    scale_x, scale_y = width / mask_w, height / mask_h

    refined, kept = [], 0
    for index, row in enumerate(rows):
        ys, xs = np.nonzero(masks[index]) if index < len(masks) else ((), ())
        if len(xs) == 0:
            refined.append(row)
            kept += 1
            continue
        old = to_xyxy(row, width, height)
        new = [xs.min() * scale_x, ys.min() * scale_y, (xs.max() + 1) * scale_x, (ys.max() + 1) * scale_y]
        old_area = (old[2] - old[0]) * (old[3] - old[1])
        area_ratio = (new[2] - new[0]) * (new[3] - new[1]) / old_area if old_area > 0 else 0.0
        if (box_iou(old, new) < args.min_iou
                or not args.min_area_ratio <= area_ratio <= args.max_area_ratio):
            refined.append(row)
            kept += 1
            continue
        refined.append((
            row[0],
            (new[0] + new[2]) / 2 / width,
            (new[1] + new[3]) / 2 / height,
            (new[2] - new[0]) / width,
            (new[3] - new[1]) / height,
        ))
    return refined, kept


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    input_yaml = input_dir / "data.yaml"
    if not input_yaml.is_file():
        sys.exit(f"ERROR: {input_yaml} not found. Run 01-export_dataset.py first.")
    data = yaml.safe_load(input_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]

    from ultralytics import SAM  # import ช้า จึงโหลดหลังตรวจ argument แล้ว

    model = SAM(args.model)
    total = refined_count = 0
    for split in ("train", "val"):
        src_images = input_dir / "images" / split
        if not src_images.is_dir():
            sys.exit(f"ERROR: Split folder not found: {src_images}")
        for kind in ("images", "labels"):
            split_dir = output_dir / kind / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for number, image_path in enumerate(image_paths, 1):
            label_name = image_path.with_suffix(".txt").name
            rows = read_labels(input_dir / "labels" / split / label_name)
            refined, kept = refine_image(model, image_path, rows, args)
            total += len(rows)
            refined_count += len(rows) - kept

            shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in refined]
            (output_dir / "labels" / split / label_name).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            print(f"\r{split}: {number}/{len(image_paths)}", end="", flush=True)
        print()

    (output_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    data_yaml = (
        f"path: {json.dumps(output_dir.as_posix(), ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(names)}\n"
        f"names: {json.dumps(names, ensure_ascii=False)}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"Boxes: {total}, refined by SAM: {refined_count}, kept original: {total - refined_count}")
    print(f"Dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
