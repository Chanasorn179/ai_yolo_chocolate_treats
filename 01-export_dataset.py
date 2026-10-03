import argparse
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = SCRIPT_DIR / "frame" / "images"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset"
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Label Studio JSON export to a YOLO detection dataset."
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        help="Label Studio JSON export (default: the only JSON file beside this script)",
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", type=float, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not 0.0 < args.train_split < 1.0:
        parser.error("--train-split must be greater than 0 and less than 1")
    return args


def find_json_file(folder: Path) -> Path:
    """Find an unambiguous Label Studio JSON export beside the script."""
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        sys.exit(
            f"ERROR: No .json file found in: {folder}\n"
            "Place the Label Studio JSON export beside this script or pass --json."
        )
    if len(json_files) > 1:
        names = "\n".join(f"  - {path.name}" for path in json_files)
        sys.exit(
            "ERROR: More than one .json file was found. Choose one with --json:\n"
            f"{names}"
        )
    return json_files[0]


def get_image_filename(task):
    """Extract the image basename from common Label Studio data formats."""
    data = task.get("data", {})
    image_value = next(
        (data[key] for key in ("image", "img", "picture", "photo") if key in data),
        None,
    )

    if image_value is None:
        image_value = next(
            (
                value
                for key, value in data.items()
                if isinstance(value, str)
                and ("image" in key.lower() or "/data/local-files" in value)
            ),
            None,
        )
    if not isinstance(image_value, str) or not image_value.strip():
        return None

    parsed = urlparse(image_value)
    local_path = parse_qs(parsed.query).get("d", [None])[0]
    path_text = unquote(local_path if local_path is not None else parsed.path)
    return Path(path_text.replace("\\", "/")).name or None


# Label Studio prefixes drag-and-drop uploads (/data/upload/...) with "<8 hex>-".
UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-")


def resolve_image_file(images_dir: Path, filename):
    """Find the source image, falling back to the name without the upload prefix."""
    if not filename:
        return None, None
    for name in (filename, UPLOAD_PREFIX_RE.sub("", filename, count=1)):
        if (images_dir / name).is_file():
            return name, images_dir / name
    return filename, None


def iter_rectangle_results(task):
    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue
        for result in annotation.get("result", []):
            if result.get("type") == "rectanglelabels":
                yield result


def collect_classes(tasks):
    """Build a deterministic class list from usable rectangle annotations."""
    classes = set()
    for task in tasks:
        for result in iter_rectangle_results(task):
            classes.update(result.get("value", {}).get("rectanglelabels", []))
    return sorted(classes)


def axis_aligned_box(result):
    """Return a clipped axis-aligned box for a Label Studio rotated rectangle.

    Label Studio stores rotation clockwise around (x, y), the rectangle's
    top-left corner. Standard YOLO Detect labels cannot retain rotation, so the
    smallest axis-aligned rectangle containing all four corners is exported.
    Rotation is calculated in pixel space because percentage x/y axes have
    different scales when the image is not square. The returned coordinates
    are percentages of the image size.
    """
    value = result.get("value", {})
    try:
        x, y, width, height = (
            float(value[key]) for key in ("x", "y", "width", "height")
        )
        rotation = float(value.get("rotation", 0) or 0)
        image_width = float(result["original_width"])
        image_height = float(result["original_height"])
    except (KeyError, TypeError, ValueError):
        return None

    numbers = (x, y, width, height, rotation, image_width, image_height)
    if not all(math.isfinite(number) for number in numbers):
        return None
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None

    x_px = x * image_width / 100.0
    y_px = y * image_height / 100.0
    width_px = width * image_width / 100.0
    height_px = height * image_height / 100.0
    angle = math.radians(rotation)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    width_vector = (width_px * cos_angle, width_px * sin_angle)
    height_vector = (-height_px * sin_angle, height_px * cos_angle)
    corners = (
        (x_px, y_px),
        (x_px + width_vector[0], y_px + width_vector[1]),
        (x_px + height_vector[0], y_px + height_vector[1]),
        (
            x_px + width_vector[0] + height_vector[0],
            y_px + width_vector[1] + height_vector[1],
        ),
    )

    left = max(0.0, min(point[0] for point in corners)) / image_width * 100.0
    top = max(0.0, min(point[1] for point in corners)) / image_height * 100.0
    right = min(image_width, max(point[0] for point in corners)) / image_width * 100.0
    bottom = min(image_height, max(point[1] for point in corners)) / image_height * 100.0
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def convert_task_to_yolo_lines(task, class_to_id):
    """Convert one Label Studio task into standard YOLO Detect labels."""
    lines = []
    skipped_boxes = 0
    for result in iter_rectangle_results(task):
        value = result.get("value", {})
        labels = value.get("rectanglelabels", [])
        box = axis_aligned_box(result)
        if not labels or box is None:
            skipped_boxes += 1
            continue

        left, top, right, bottom = box
        x_center = (left + right) / 200.0
        y_center = (top + bottom) / 200.0
        width = (right - left) / 100.0
        height = (bottom - top) / 100.0

        for label in labels:
            if label not in class_to_id:
                skipped_boxes += 1
                continue
            lines.append(
                f"{class_to_id[label]} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )
    return lines, skipped_boxes


def clean_output_splits(output_dir: Path):
    """Remove prior generated splits so stale files cannot leak across runs."""
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            split_dir = output_dir / kind / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    json_path = (args.json_path or find_json_file(SCRIPT_DIR)).resolve()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not json_path.is_file():
        sys.exit(f"ERROR: JSON file not found: {json_path}")
    if not images_dir.is_dir():
        sys.exit(f"ERROR: Source images folder not found: {images_dir}")

    print(f"Using JSON file: {json_path}")
    try:
        tasks = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"ERROR: Could not read JSON export: {error}")
    if not isinstance(tasks, list):
        sys.exit("ERROR: Expected a Label Studio JSON export containing a list of tasks.")

    candidate_tasks = []
    skipped_no_file = 0
    for task in tasks:
        filename, src_image_path = resolve_image_file(images_dir, get_image_filename(task))
        if src_image_path is None:
            skipped_no_file += 1
            continue
        candidate_tasks.append((task, filename, src_image_path))

    classes = collect_classes(task for task, _, _ in candidate_tasks)
    if not classes:
        sys.exit("ERROR: No usable rectanglelabels annotations were found.")
    class_to_id = {name: index for index, name in enumerate(classes)}

    samples = []
    skipped_no_boxes = 0
    skipped_boxes = 0
    seen_filenames = set()
    for task, filename, src_image_path in candidate_tasks:
        normalized_name = filename.casefold()
        if normalized_name in seen_filenames:
            sys.exit(f"ERROR: Duplicate output image filename in JSON export: {filename}")
        seen_filenames.add(normalized_name)

        lines, invalid_count = convert_task_to_yolo_lines(task, class_to_id)
        skipped_boxes += invalid_count
        if not lines:
            skipped_no_boxes += 1
            continue
        samples.append((filename, src_image_path, lines))

    if not samples:
        sys.exit("ERROR: No images with valid bounding boxes could be converted.")

    random.Random(args.seed).shuffle(samples)
    if len(samples) == 1:
        split_index = 1
    else:
        split_index = round(len(samples) * args.train_split)
        split_index = min(max(split_index, 1), len(samples) - 1)
    split_samples = {
        "train": samples[:split_index],
        "val": samples[split_index:],
    }

    clean_output_splits(output_dir)
    for split_name, items in split_samples.items():
        for filename, src_image_path, lines in items:
            shutil.copy2(src_image_path, output_dir / "images" / split_name / filename)
            label_path = (output_dir / "labels" / split_name / filename).with_suffix(".txt")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (output_dir / "classes.txt").write_text(
        "\n".join(classes) + "\n", encoding="utf-8"
    )
    yaml_path = output_dir.resolve().as_posix()
    data_yaml = (
        f"path: {json.dumps(yaml_path, ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(classes)}\n"
        f"names: {json.dumps(classes, ensure_ascii=False)}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    if skipped_no_file:
        print(f"WARNING: Skipped {skipped_no_file} tasks whose image file was not found.")
    if skipped_no_boxes:
        print(f"WARNING: Skipped {skipped_no_boxes} tasks with no valid bounding boxes.")
    if skipped_boxes:
        print(f"WARNING: Skipped {skipped_boxes} invalid bounding-box results.")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {len(split_samples['train'])} images")
    print(f"Val:   {len(split_samples['val'])} images")
    print(f"Dataset written to: {output_dir}")
    print("Ready to train with Ultralytics, for example:")
    print(f'  yolo detect train data="{output_dir / "data.yaml"}" model=yolov8n.pt epochs=100')


if __name__ == "__main__":
    main()
