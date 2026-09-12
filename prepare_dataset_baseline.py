import os
import json
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(
    "/Volumes/AIR_DISK/training-data/object-detection-YCB"
)

TRAIN_REAL_DIR = BASE_DIR / "train_real"
OUTPUT_DIR = BASE_DIR / "classification_dataset"

CLASS_NAMES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

# ============================================================
# Dataset preparation settings
# ============================================================

# 1  = use every frame
# 5  = use every 5th frame
# 10 = use every 10th frame
#
# 建议第一次先设成 10，确认代码没问题后再改小。
FRAME_STEP = 10

# 如果裁出来的物体太小，就跳过
MIN_CROP_SIZE = 30

# 在物体 bounding box 周围额外保留一点背景
PADDING = 10


# ============================================================
# Check paths
# ============================================================

if not TRAIN_REAL_DIR.exists():
    raise FileNotFoundError(
        f"Cannot find train_real directory:\n{TRAIN_REAL_DIR}"
    )

print("Train directory:")
print(TRAIN_REAL_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# Create 21 class folders
# ============================================================

for class_name in CLASS_NAMES.values():
    class_dir = OUTPUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Find scene folders
# Example:
# train_real/000000
# train_real/000001
# ...
# ============================================================

scene_dirs = sorted(
    [
        path
        for path in TRAIN_REAL_DIR.iterdir()
        if path.is_dir()
    ]
)

print(f"\nFound {len(scene_dirs)} scenes.")


# ============================================================
# Counters
# ============================================================

total_saved = 0
total_frames_processed = 0
total_objects_seen = 0

saved_per_class = {
    obj_id: 0
    for obj_id in CLASS_NAMES
}


# ============================================================
# Process all scenes
# ============================================================

for scene_number, scene_dir in enumerate(scene_dirs, start=1):

    scene_name = scene_dir.name

    print(
        f"\n[{scene_number}/{len(scene_dirs)}] "
        f"Processing scene {scene_name}"
    )

    rgb_dir = scene_dir / "rgb"
    mask_visib_dir = scene_dir / "mask_visib"
    gt_path = scene_dir / "scene_gt.json"

    if not rgb_dir.exists():
        print("  Missing rgb folder. Skipping.")
        continue

    if not mask_visib_dir.exists():
        print("  Missing mask_visib folder. Skipping.")
        continue

    if not gt_path.exists():
        print("  Missing scene_gt.json. Skipping.")
        continue

    # --------------------------------------------------------
    # Load scene_gt.json
    # --------------------------------------------------------

    with open(gt_path, "r") as file:
        scene_gt = json.load(file)

    # JSON keys are strings, so sort numerically
    frame_ids = sorted(
        scene_gt.keys(),
        key=lambda x: int(x)
    )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    for frame_index, frame_id in enumerate(frame_ids):

        # Use only every FRAME_STEP frame
        if frame_index % FRAME_STEP != 0:
            continue

        frame_number = int(frame_id)

        rgb_path = (
            rgb_dir
            / f"{frame_number:06d}.png"
        )

        if not rgb_path.exists():
            continue

        # Read RGB image
        image = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            print(
                f"  Failed to read image: {rgb_path}"
            )
            continue

        total_frames_processed += 1

        image_height, image_width = image.shape[:2]

        # Each frame can contain multiple objects
        objects = scene_gt[frame_id]

        for object_index, object_info in enumerate(objects):

            total_objects_seen += 1

            obj_id = object_info["obj_id"]

            if obj_id not in CLASS_NAMES:
                continue

            class_name = CLASS_NAMES[obj_id]

            # ------------------------------------------------
            # BOP mask filename format:
            #
            # frame_id_object_index.png
            #
            # Example:
            # 000123_000000.png
            # ------------------------------------------------

            mask_path = (
                mask_visib_dir
                / (
                    f"{frame_number:06d}_"
                    f"{object_index:06d}.png"
                )
            )

            if not mask_path.exists():
                continue

            # Read visible-object mask
            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:
                continue

            # ------------------------------------------------
            # Find all visible pixels belonging to this object
            # ------------------------------------------------

            y_coordinates, x_coordinates = np.where(
                mask > 0
            )

            if (
                len(x_coordinates) == 0
                or len(y_coordinates) == 0
            ):
                continue

            # Bounding box
            x_min = int(x_coordinates.min())
            x_max = int(x_coordinates.max())

            y_min = int(y_coordinates.min())
            y_max = int(y_coordinates.max())

            # ------------------------------------------------
            # Add padding
            # ------------------------------------------------

            x_min = max(
                0,
                x_min - PADDING
            )

            y_min = max(
                0,
                y_min - PADDING
            )

            x_max = min(
                image_width - 1,
                x_max + PADDING
            )

            y_max = min(
                image_height - 1,
                y_max + PADDING
            )

            crop_width = x_max - x_min + 1
            crop_height = y_max - y_min + 1

            # ------------------------------------------------
            # Skip objects that are too small
            # ------------------------------------------------

            if (
                crop_width < MIN_CROP_SIZE
                or crop_height < MIN_CROP_SIZE
            ):
                continue

            # ------------------------------------------------
            # Crop RGB object
            # ------------------------------------------------

            cropped_image = image[
                y_min:y_max + 1,
                x_min:x_max + 1
            ]

            if cropped_image.size == 0:
                continue

            # ------------------------------------------------
            # Output filename
            #
            # scene_frame_objectindex.png
            # ------------------------------------------------

            output_filename = (
                f"{scene_name}_"
                f"{frame_number:06d}_"
                f"{object_index:02d}.png"
            )

            output_path = (
                OUTPUT_DIR
                / class_name
                / output_filename
            )

            # ------------------------------------------------
            # Save cropped object image
            # ------------------------------------------------

            success = cv2.imwrite(
                str(output_path),
                cropped_image
            )

            if success:
                total_saved += 1
                saved_per_class[obj_id] += 1


# ============================================================
# Final summary
# ============================================================

print("\n")
print("=" * 60)
print("Dataset preparation finished")
print("=" * 60)

print(
    f"Frames processed: {total_frames_processed}"
)

print(
    f"Objects examined: {total_objects_seen}"
)

print(
    f"Images saved: {total_saved}"
)

print("\nImages per class:")
print("-" * 60)

for obj_id in sorted(CLASS_NAMES):

    class_name = CLASS_NAMES[obj_id]

    count = saved_per_class[obj_id]

    print(
        f"{obj_id:02d} "
        f"{class_name:<28} "
        f"{count}"
    )

print("-" * 60)

print("\nDataset saved to:")
print(OUTPUT_DIR)
