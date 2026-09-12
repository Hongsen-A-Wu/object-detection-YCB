from pathlib import Path
import shutil


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(
    "/Volumes/AIR_DISK/training-data/object-detection-YCB"
)

SOURCE_DIR = BASE_DIR / "classification_dataset"

OUTPUT_DIR = BASE_DIR / "classification_dataset_split"

TRAIN_DIR = OUTPUT_DIR / "train"
VALIDATION_DIR = OUTPUT_DIR / "validation"
TEST_DIR = OUTPUT_DIR / "test"


# ============================================================
# Scene split
# ============================================================

# 000000 ~ 000063
TRAIN_SCENES = set(range(0, 64))

# 000064 ~ 000071
VALIDATION_SCENES = set(range(64, 72))

# 000072 ~ 000079
TEST_SCENES = set(range(72, 80))


# ============================================================
# Check source directory
# ============================================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Cannot find source dataset:\n{SOURCE_DIR}"
    )


# ============================================================
# Create output folders
# ============================================================

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Counters
# ============================================================

train_count = 0
validation_count = 0
test_count = 0

train_per_class = {}
validation_per_class = {}
test_per_class = {}


# ============================================================
# Go through all class folders
# ============================================================

class_dirs = sorted(
    [
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_dir()
    ]
)

print(f"Found {len(class_dirs)} classes.")


for class_dir in class_dirs:

    class_name = class_dir.name

    print(f"\nProcessing class: {class_name}")

    # Create class folders
    train_class_dir = TRAIN_DIR / class_name
    validation_class_dir = VALIDATION_DIR / class_name
    test_class_dir = TEST_DIR / class_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    validation_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    train_per_class[class_name] = 0
    validation_per_class[class_name] = 0
    test_per_class[class_name] = 0

    image_files = sorted(
        class_dir.glob("*.png")
    )

    for image_path in image_files:

        filename = image_path.name

        # Example:
        # 000023_001250_01.png
        #
        # First 6 digits = scene number

        try:
            scene_number = int(
                filename.split("_")[0]
            )

        except ValueError:
            print(
                f"Skipping invalid filename: {filename}"
            )
            continue

        # ====================================================
        # Decide destination
        # ====================================================

        if scene_number in TRAIN_SCENES:

            destination = (
                train_class_dir
                / filename
            )

            train_count += 1
            train_per_class[class_name] += 1

        elif scene_number in VALIDATION_SCENES:

            destination = (
                validation_class_dir
                / filename
            )

            validation_count += 1
            validation_per_class[class_name] += 1

        elif scene_number in TEST_SCENES:

            destination = (
                test_class_dir
                / filename
            )

            test_count += 1
            test_per_class[class_name] += 1

        else:

            print(
                f"Unknown scene number: "
                f"{scene_number}"
            )

            continue

        # ====================================================
        # Copy file
        # ====================================================

        shutil.copy2(
            image_path,
            destination
        )


# ============================================================
# Summary
# ============================================================

total_count = (
    train_count
    + validation_count
    + test_count
)

print("\n")
print("=" * 70)
print("Dataset split finished")
print("=" * 70)

print(f"Train images:      {train_count}")
print(f"Validation images: {validation_count}")
print(f"Test images:       {test_count}")
print(f"Total images:      {total_count}")

print("\n")

if total_count > 0:

    print(
        f"Train:      "
        f"{train_count / total_count * 100:.2f}%"
    )

    print(
        f"Validation: "
        f"{validation_count / total_count * 100:.2f}%"
    )

    print(
        f"Test:       "
        f"{test_count / total_count * 100:.2f}%"
    )


# ============================================================
# Per-class summary
# ============================================================

print("\n")
print("=" * 70)
print("Images per class")
print("=" * 70)

print(
    f"{'Class':<30}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
)

print("-" * 70)

for class_name in sorted(train_per_class):

    print(
        f"{class_name:<30}"
        f"{train_per_class[class_name]:>10}"
        f"{validation_per_class[class_name]:>10}"
        f"{test_per_class[class_name]:>10}"
    )


print("\nDataset saved to:")
print(OUTPUT_DIR)