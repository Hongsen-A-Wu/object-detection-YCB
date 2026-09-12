from pathlib import Path
import json
import cv2


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(
    "/Volumes/AIR_DISK/training-data/object-detection-YCB"
)

# Official BOP YCB-V test dataset
TEST_SOURCE_DIR = BASE_DIR / "test"

# Output classification test dataset
OUTPUT_DIR = BASE_DIR / "classification_test_dataset"


# ============================================================
# YCB-V class names
# ============================================================

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
# Settings
# ============================================================

# Ignore objects that are almost completely hidden
MIN_VISIBLE_FRACTION = 0.1

# Ignore extremely tiny crops
MIN_WIDTH = 20
MIN_HEIGHT = 20

# Add some context around bounding box
PADDING_RATIO = 0.10


# ============================================================
# Check source directory
# ============================================================

if not TEST_SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Cannot find official test dataset:\n"
        f"{TEST_SOURCE_DIR}"
    )


# ============================================================
# Create output folders
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for class_name in CLASS_NAMES.values():
    class_dir = OUTPUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Statistics
# ============================================================

images_saved_per_class = {
    class_name: 0
    for class_name in CLASS_NAMES.values()
}

frames_processed = 0
objects_examined = 0
images_saved = 0
objects_skipped = 0


# ============================================================
# Process each test scene
# ============================================================

scene_dirs = sorted(
    [
        path
        for path in TEST_SOURCE_DIR.iterdir()
        if path.is_dir()
    ]
)


print("=" * 60)
print("Preparing official YCB-V test dataset")
print("=" * 60)

print(f"Source: {TEST_SOURCE_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Scenes found: {len(scene_dirs)}")
print()


for scene_dir in scene_dirs:

    scene_name = scene_dir.name

    rgb_dir = scene_dir / "rgb"
    gt_path = scene_dir / "scene_gt.json"
    gt_info_path = scene_dir / "scene_gt_info.json"

    # --------------------------------------------------------
    # Check scene files
    # --------------------------------------------------------

    if not rgb_dir.exists():
        print(
            f"Skipping {scene_name}: "
            f"cannot find rgb folder"
        )
        continue

    if not gt_path.exists():
        print(
            f"Skipping {scene_name}: "
            f"cannot find scene_gt.json"
        )
        continue

    if not gt_info_path.exists():
        print(
            f"Skipping {scene_name}: "
            f"cannot find scene_gt_info.json"
        )
        continue

    print(f"Processing scene {scene_name}...")


    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    with open(gt_path, "r") as file:
        scene_gt = json.load(file)

    with open(gt_info_path, "r") as file:
        scene_gt_info = json.load(file)


    # --------------------------------------------------------
    # Process each frame
    # --------------------------------------------------------

    for frame_id in scene_gt:

        frames_processed += 1

        frame_number = int(frame_id)

        image_path_png = (
            rgb_dir /
            f"{frame_number:06d}.png"
        )

        image_path_jpg = (
            rgb_dir /
            f"{frame_number:06d}.jpg"
        )


        # ----------------------------------------------------
        # Find RGB image
        # ----------------------------------------------------

        if image_path_png.exists():
            image_path = image_path_png

        elif image_path_jpg.exists():
            image_path = image_path_jpg

        else:
            print(
                f"Image not found: "
                f"scene {scene_name}, "
                f"frame {frame_number:06d}"
            )
            continue


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Failed to read: {image_path}"
            )
            continue

        image_height, image_width = image.shape[:2]


        # ----------------------------------------------------
        # Object annotations
        # ----------------------------------------------------

        objects_gt = scene_gt[frame_id]
        objects_info = scene_gt_info[frame_id]


        # scene_gt and scene_gt_info use the same
        # object ordering.
        for object_index, (
            gt,
            info
        ) in enumerate(
            zip(objects_gt, objects_info)
        ):

            objects_examined += 1


            # =================================================
            # Object ID
            # =================================================

            obj_id = gt["obj_id"]

            if obj_id not in CLASS_NAMES:
                objects_skipped += 1
                continue

            class_name = CLASS_NAMES[obj_id]


            # =================================================
            # Visibility
            # =================================================

            visible_fraction = info.get(
                "visib_fract",
                1.0
            )

            if visible_fraction < MIN_VISIBLE_FRACTION:
                objects_skipped += 1
                continue


            # =================================================
            # Bounding box
            # =================================================

            # BOP format:
            #
            # bbox_visib = [x, y, width, height]
            #
            bbox = info.get("bbox_visib")

            if bbox is None:
                objects_skipped += 1
                continue

            x, y, width, height = bbox


            # =================================================
            # Check bbox size
            # =================================================

            if (
                width < MIN_WIDTH
                or
                height < MIN_HEIGHT
            ):
                objects_skipped += 1
                continue


            # =================================================
            # Add padding
            # =================================================

            padding_x = int(
                width * PADDING_RATIO
            )

            padding_y = int(
                height * PADDING_RATIO
            )


            x1 = max(
                0,
                int(x - padding_x)
            )

            y1 = max(
                0,
                int(y - padding_y)
            )

            x2 = min(
                image_width,
                int(x + width + padding_x)
            )

            y2 = min(
                image_height,
                int(y + height + padding_y)
            )


            # =================================================
            # Check crop
            # =================================================

            if x2 <= x1 or y2 <= y1:
                objects_skipped += 1
                continue


            # =================================================
            # Crop object
            # =================================================

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                objects_skipped += 1
                continue


            # =================================================
            # Save image
            # =================================================

            output_class_dir = (
                OUTPUT_DIR /
                class_name
            )

            output_filename = (
                f"scene_{scene_name}_"
                f"frame_{frame_number:06d}_"
                f"obj_{object_index:02d}.jpg"
            )

            output_path = (
                output_class_dir /
                output_filename
            )

            success = cv2.imwrite(
                str(output_path),
                crop
            )

            if success:
                images_saved += 1
                images_saved_per_class[
                    class_name
                ] += 1

            else:
                print(
                    f"Failed to save: "
                    f"{output_path}"
                )


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 60)
print("Official test dataset preparation finished")
print("=" * 60)

print(
    f"Scenes processed: "
    f"{len(scene_dirs)}"
)

print(
    f"Frames processed: "
    f"{frames_processed}"
)

print(
    f"Objects examined: "
    f"{objects_examined}"
)

print(
    f"Images saved: "
    f"{images_saved}"
)

print(
    f"Objects skipped: "
    f"{objects_skipped}"
)


print()
print("Images per class:")
print("-" * 60)


for obj_id, class_name in CLASS_NAMES.items():

    count = images_saved_per_class[
        class_name
    ]

    print(
        f"{obj_id:02d} "
        f"{class_name:<30} "
        f"{count}"
    )


print()
print(
    "Output dataset:"
)

print(
    OUTPUT_DIR
)

print("=" * 60)