from pathlib import Path
import shutil
from collections import Counter


# Edit this dictionary to match your dataset labels
LABEL_TO_DIRNAME = {
    1: 'Surprise',
    2: 'Fear',
    3: 'Disgust',
    4: 'Happy',
    5: 'Sad',
    6: 'Angry',
    7: 'Neutral',
}


def split_images_by_label(
    images_dir,
    labels_txt_path,
    output_dir=None,
    label_to_dirname=None,
    copy_files=True,
    allowed_extensions=(".jpg", ".jpeg", ".JPG", ".JPEG"),
):
    """
    Create one directory per class and copy/move images into it according to a label text file.

    Expected label file format:
        image_filename.jpg label_index

    Example:
        test_0001_aligned.jpg 5
        test_0002_aligned.jpg 1
    """

    images_dir = Path(images_dir)
    labels_txt_path = Path(labels_txt_path)

    if output_dir is None:
        output_dir = images_dir.parent / f"{images_dir.name}_by_label"
    else:
        output_dir = Path(output_dir)

    if label_to_dirname is None:
        label_to_dirname = LABEL_TO_DIRNAME

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_txt_path.is_file():
        raise FileNotFoundError(f"Label file not found: {labels_txt_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the 7 class directories
    for label_idx, dirname in label_to_dirname.items():
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    missing_images = []
    malformed_lines = []
    unknown_labels = []

    with labels_txt_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                malformed_lines.append((line_number, line))
                continue

            filename, label_str = parts

            try:
                label_idx = int(label_str)
            except ValueError:
                malformed_lines.append((line_number, line))
                continue

            if label_idx not in label_to_dirname:
                unknown_labels.append((line_number, filename, label_idx))
                continue

            if not filename.endswith(allowed_extensions):
                malformed_lines.append((line_number, line))
                continue

            src = images_dir / filename.replace('_aligned', '')

            if not src.is_file():
                missing_images.append(filename)
                continue

            class_dir = output_dir / label_to_dirname[label_idx]
            dst = class_dir / filename

            if copy_files:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

            counts[label_idx] += 1

    print(f"Done. Output directory: {output_dir}")
    print("\nImages per class:")
    for label_idx, dirname in label_to_dirname.items():
        print(f"  {label_idx} -> {dirname}: {counts[label_idx]}")

    if missing_images:
        print(f"\nMissing images: {len(missing_images)}")
        print("First few missing:", missing_images[:10])

    if malformed_lines:
        print(f"\nMalformed lines: {len(malformed_lines)}")
        print("First few malformed:", malformed_lines[:10])

    if unknown_labels:
        print(f"\nUnknown labels: {len(unknown_labels)}")
        print("First few unknown:", unknown_labels[:10])


if __name__ == "__main__":
    split_images_by_label(
        images_dir="/home/projects/bagon/ilanaveh/data/RAF-DB/basic/Image/aligned_224",
        labels_txt_path="/home/projects/bagon/ilanaveh/data/RAF-DB/basic/EmoLabel/test.txt",
        output_dir="/home/projects/bagon/ilanaveh/data/RAF-DB/split_test",
        label_to_dirname=LABEL_TO_DIRNAME,
        copy_files=True,   # True = copy images, False = move images
    )