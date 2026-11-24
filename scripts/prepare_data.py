"""
IT8414 Tomato Leaf Classification - data preparation script
Author: Fatema Alashiri

Rebuilds the project dataset from the original Kaggle images without the
notebook. It performs exactly the same steps, with the same seed, as notebook
cells 2.0-4.0, so it produces the same 300 images and the same split.

Steps
  1. Count the images in the three selected class folders.
  2. Copy a balanced, seeded random subset of 100 images per class.
  3. Check every selected image opens, and record sizes and colour modes.
  4. Stratified 80 / 10 / 10 train / validation / test split (seed 42). The ratio
     was chosen by the split-ratio experiment in notebook Cell 5.3 (--ratio to change).
  5. Leakage checks: exact file overlap and near-duplicate images (difference hash).
  6. Save portable split manifests (CSV) and a JSON summary.

Usage (from the project root)
  python scripts/prepare_data.py
  python scripts/prepare_data.py --original data/original --project . --rebuild

Requirements: numpy, pandas, pillow, scikit-learn (no TensorFlow needed).
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42
IMAGES_PER_CLASS = 100
EXPECTED_CLASSES = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"]
LABEL_IDS = {"Tomato___healthy": 0, "Tomato___Early_blight": 1, "Tomato___Late_blight": 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
HAMMING_THRESHOLD = 5  # hashes differing in <= 5 of 64 bits = near-duplicate


def list_images(folder):
    return sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def count_original(original_path):
    counts = {}
    for class_name in EXPECTED_CLASSES:
        class_path = original_path / class_name
        if not class_path.exists():
            raise FileNotFoundError(f"Class folder not found: {class_path}")
        counts[class_name] = len(list_images(class_path))
    return counts


def subset_complete(selected_path):
    return selected_path.exists() and all(
        (selected_path / c).exists() and len(list_images(selected_path / c)) == IMAGES_PER_CLASS
        for c in EXPECTED_CLASSES
    )


def select_balanced_subset(original_path, selected_path, rebuild):
    if subset_complete(selected_path) and not rebuild:
        print("Balanced subset already exists - keeping it.")
        return
    # Remove only the class folders, so data/selected/README.md is kept
    for class_name in EXPECTED_CLASSES:
        if (selected_path / class_name).exists():
            shutil.rmtree(selected_path / class_name)
    selected_path.mkdir(parents=True, exist_ok=True)

    # One random generator used across the classes in a fixed order (same as the notebook)
    selection_rng = random.Random(SEED)
    for class_name in EXPECTED_CLASSES:
        destination = selected_path / class_name
        destination.mkdir(exist_ok=True)
        chosen = selection_rng.sample(list_images(original_path / class_name), IMAGES_PER_CLASS)
        for image_path in chosen:
            shutil.copy2(image_path, destination / image_path.name)
        print(f"{class_name}: {len(chosen)} images selected")


def build_table(selected_path):
    records = [
        {"image_path": str(p), "label": c}
        for c in EXPECTED_CLASSES
        for p in list_images(selected_path / c)
    ]
    return pd.DataFrame(records)


def quality_check(data_df):
    sizes, modes, corrupted = set(), set(), []
    for image_path in data_df["image_path"]:
        try:
            with Image.open(image_path) as img:
                sizes.add(img.size)
                modes.add(img.mode)
                img.verify()
        except Exception:
            corrupted.append(image_path)
    return sorted(sizes), sorted(modes), corrupted


def stratified_split(data_df, ratio=(0.80, 0.10, 0.10)):
    # Same two-step split as make_split() in notebook Cell 4.0
    train_share, val_share, test_share = ratio
    train_df, temp_df = train_test_split(
        data_df, train_size=train_share, stratify=data_df["label"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_share / (val_share + test_share),
        stratify=temp_df["label"], random_state=SEED)
    return {"train": train_df.reset_index(drop=True), "validation": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True)}


def difference_hash(image_path, hash_size=8):
    with Image.open(image_path) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def leakage_checks(splits):
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    exact = {
        f"{a}/{b}": len(set(splits[a]["image_path"]) & set(splits[b]["image_path"]))
        for a, b in pairs
    }
    hashes = {s: np.array([difference_hash(p) for p in f["image_path"]]) for s, f in splits.items()}
    near, examples = {}, []
    for a, b in pairs:
        distances = (hashes[a][:, None, :] != hashes[b][None, :, :]).sum(axis=2)
        found = np.argwhere(distances <= HAMMING_THRESHOLD)
        near[f"{a}/{b}"] = int(len(found))
        for i, j in found[:10]:
            examples.append([splits[a]["image_path"].iloc[i], splits[b]["image_path"].iloc[j],
                             int(distances[i, j])])
    return exact, near, examples


def manifest_table(frame, split_name):
    return pd.DataFrame({
        "filename": [Path(p).name for p in frame["image_path"]],
        "label": frame["label"].values,
        "label_id": frame["label"].map(LABEL_IDS).values,
        "split": split_name,
        "relative_path": [f"data/selected/{lab}/{Path(p).name}"
                          for lab, p in zip(frame["label"], frame["image_path"])],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".", help="project root folder (default: current folder)")
    parser.add_argument("--original", default=None, help="folder holding the original class folders "
                                                          "(default: <project>/data/original)")
    parser.add_argument("--rebuild", action="store_true", help="recreate the balanced subset from scratch")
    parser.add_argument("--ratio", default="80/10/10", help="train/validation/test ratio (default 80/10/10, "
                                                           "chosen in notebook Cell 5.3)")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    original_path = Path(args.original).resolve() if args.original else project / "data" / "original"
    selected_path = project / "data" / "selected"
    split_path = project / "data" / "split_manifests"
    split_path.mkdir(parents=True, exist_ok=True)

    print("1. Original class counts")
    counts = count_original(original_path)
    total = sum(counts.values())
    for c, n in counts.items():
        print(f"   {c}: {n} ({n / total:.1%})")
    print(f"   Imbalance ratio: {max(counts.values()) / min(counts.values()):.2f}")

    print("2. Balanced subset")
    select_balanced_subset(original_path, selected_path, args.rebuild)
    data_df = build_table(selected_path)

    print("3. Quality check")
    sizes, modes, corrupted = quality_check(data_df)
    print(f"   Sizes: {sizes} | Modes: {modes} | Corrupted: {len(corrupted)}")

    print("4. Stratified split")
    ratio = tuple(int(x) / 100 for x in args.ratio.split("/"))
    splits = stratified_split(data_df, ratio)
    for s, f in splits.items():
        print(f"   {s}: {len(f)} -> {f['label'].value_counts().to_dict()}")

    print("5. Leakage checks")
    exact, near, examples = leakage_checks(splits)
    print(f"   Exact overlap: {exact}")
    print(f"   Near-duplicates (<= {HAMMING_THRESHOLD} of 64 bits): {near}")

    print("6. Saving manifests")
    file_names = {"train": "train_split.csv", "validation": "validation_split.csv", "test": "test_split.csv"}
    for s, f in splits.items():
        manifest_table(f, s).to_csv(split_path / file_names[s], index=False)

    summary = {
        "seed": SEED,
        "original_counts": counts,
        "images_per_class": IMAGES_PER_CLASS,
        "image_sizes": [list(x) for x in sizes],
        "colour_modes": modes,
        "corrupted_images": len(corrupted),
        "split_ratio": args.ratio,
        "split_sizes": {s: len(f) for s, f in splits.items()},
        "split_class_counts": {s: f["label"].value_counts().to_dict() for s, f in splits.items()},
        "exact_overlap": exact,
        "near_duplicate_pairs": near,
        "near_duplicate_examples": examples,
    }
    (split_path / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"   Saved to {split_path}")


if __name__ == "__main__":
    main()
