import os
import json

from vit_multimodal import config

from captioning.generate_attributes import (
    get_attributes_batch
)


BATCH_SIZE = 32


train_folder = config.TRAIN_DIR

if train_folder is None:

    raise ValueError(
        f"{config.DATASET} does not have a training set."
    )

output_dir = os.path.dirname(
    config.TRAIN_CAPTIONS
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_file = config.TRAIN_CAPTIONS

print(f"Dataset      : {config.DATASET}")
print(f"Train Folder : {train_folder}")

if not os.path.exists(train_folder):

    raise FileNotFoundError(
        f"{train_folder} not found."
    )

images = []

for root, _, files in os.walk(train_folder):

    for img in sorted(files):

        if img.lower().endswith(
            config.IMAGE_EXTENSIONS
        ):

            images.append(
                os.path.join(root, img)
            )

print(f"Total Images : {len(images)}")

if os.path.exists(output_file):

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        captions = json.load(f)

    print(
        f"Loaded {len(captions)} existing captions."
    )

else:

    captions = {}

    print("Starting fresh.")

remaining_images = []

for path in images:

    if config.DATASET in [
        "P-DukeMTMC-reid",
        "Occluded_REID"
    ]:

        img_key = os.path.relpath(
            path,
            train_folder
        ).replace("\\", "/")

    else:

        img_key = os.path.basename(path)

    if img_key not in captions:

        remaining_images.append(
            (path, img_key)
        )

print(f"Remaining Images : {len(remaining_images)}")

total_batches = (
    len(remaining_images)
    + BATCH_SIZE
    - 1
) // BATCH_SIZE

for batch_idx in range(total_batches):

    batch = remaining_images[
        batch_idx * BATCH_SIZE:
        (batch_idx + 1) * BATCH_SIZE
    ]

    batch_paths = [
        p[0]
        for p in batch
    ]

    print(
        f"\nProcessing Batch "
        f"{batch_idx + 1}/{total_batches}"
    )

    try:

        batch_results = get_attributes_batch(
            batch_paths
        )

    except Exception as e:

        print(e)

        continue

    for path, img_key in batch:

        if path in batch_results:

            captions[img_key] = batch_results[path]

            print(f"✓ {img_key}")

    if len(captions) % 250 < BATCH_SIZE:

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                captions,
                f,
                indent=4
            )

        print(
            f"\nSaved {len(captions)} captions.\n"
        )

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        captions,
        f,
        indent=4
    )

print("\n================================")
print("Caption Generation Finished")
print("================================\n")

print(f"Saved : {output_file}")
print(f"Total Captions : {len(captions)}")