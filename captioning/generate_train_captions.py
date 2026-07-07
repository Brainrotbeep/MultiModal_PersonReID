import os
import json

from captioning.generate_attributes import get_attributes


# project root

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

train_folder = os.path.join(
    BASE_DIR,
    "data",
    "Market-1501",
    "bounding_box_train"
)

output_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "captions"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_file = os.path.join(
    output_dir,
    "train_captions.json"
)

print("Train folder:", train_folder)
print("Exists:", os.path.exists(train_folder))

images = sorted([
    img
    for img in os.listdir(train_folder)
    if img.endswith(".jpg")
])

if os.path.exists(output_file):

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        captions = json.load(f)

    print(
        f"Loaded {len(captions)} existing captions"
    )

else:

    captions = {}

    print(
        "Starting fresh"
    )

print(f"\nTotal images: {len(images)}\n")


for idx, img in enumerate(images):

    # skip images already processed

    if img in captions:
        continue

    path = os.path.join(
        train_folder,
        img
    )

    print(
        f"[{idx + 1}/{len(images)}] Processing:",
        img
    )

    attrs = get_attributes(path)

    captions[img] = attrs

    # save every 500 images

    if (idx + 1) % 500 == 0:

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
            f"\n✅ Saved progress at {len(captions)} images\n"
        )


# final save

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
    f"\n✅ Finished. Saved: {output_file}"
)