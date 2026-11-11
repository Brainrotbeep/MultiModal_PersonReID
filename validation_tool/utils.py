import os
import json
import random
import cv2

from config import (
    DATASETS,
    RANDOM_SEED,
    SAMPLES_PER_DATASET
)


random.seed(RANDOM_SEED)


# =====================================================
# Load Caption JSON
# =====================================================

def load_captions(json_file):

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =====================================================
# Valid Image Extensions
# =====================================================

IMAGE_EXTENSIONS = (

    ".jpg",

    ".jpeg",

    ".png",

    ".bmp",

    ".tif",

    ".tiff"

)


# =====================================================
# Collect Images
# =====================================================

def collect_images(image_dir):

    images = []

    for root, _, files in os.walk(image_dir):

        for file in sorted(files):

            if file.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                images.append(

                    os.path.join(
                        root,
                        file
                    )

                )

    return images


# =====================================================
# Random Sampling
# =====================================================

def random_sample(image_list):

    if len(image_list) <= SAMPLES_PER_DATASET:

        return image_list

    return random.sample(

        image_list,

        SAMPLES_PER_DATASET

    )


# =====================================================
# Caption Key Generator
# =====================================================

def get_caption_key(

    dataset_name,

    image_path,

    image_root

):

    filename = os.path.basename(
        image_path
    )

    if dataset_name in [

        "P-DukeMTMC-reid",

        "Occluded_REID"

    ]:

        return os.path.relpath(

            image_path,

            image_root

        ).replace(

            "\\",

            "/"

        )

    return filename


# =====================================================
# Image Loader
# =====================================================

def load_image(image_path):

    image = cv2.imread(

        image_path

    )

    if image is None:

        raise ValueError(

            f"Cannot read image\n{image_path}"

        )

    return image


# =====================================================
# Resize while keeping aspect ratio
# =====================================================

def resize_image(

    image,

    max_width=600,

    max_height=700

):

    h, w = image.shape[:2]

    scale = min(

        max_width / w,

        max_height / h

    )

    if scale > 1:

        scale = 1

    new_w = int(w * scale)

    new_h = int(h * scale)

    image = cv2.resize(

        image,

        (

            new_w,

            new_h

        )

    )

    return image


# =====================================================
# Draw Caption Panel
# =====================================================

def build_caption_text(attributes):

    text = [

        f"Age            : {attributes.get('age', 'Unknown')}",

        f"Gender         : {attributes.get('gender', 'Unknown')}",

        f"Upper Color    : {attributes.get('upper_color', 'Unknown')}",

        f"Upper Type     : {attributes.get('upper_type', 'Unknown')}",

        f"Lower Color    : {attributes.get('lower_color', 'Unknown')}",

        f"Lower Type     : {attributes.get('lower_type', 'Unknown')}",

        f"Footwear Color : {attributes.get('footwear_color', 'Unknown')}"

    ]

    return text


# =====================================================
# Dataset Loader
# =====================================================

def load_dataset(dataset_name):

    dataset = DATASETS[

        dataset_name

    ]

    image_dir = dataset[

        "image_dir"

    ]

    caption_file = dataset[

        "caption_file"

    ]

    captions = load_captions(

        caption_file

    )

    images = collect_images(

        image_dir

    )

    images = random_sample(

        images

    )

    samples = []

    for image_path in images:

        key = get_caption_key(

            dataset_name,

            image_path,

            image_dir

        )

        if key not in captions:

            continue

        samples.append(

            {

                "dataset": dataset_name,

                "image_path": image_path,

                "caption_key": key,

                "attributes": captions[key]

            }

        )

    return samples


# =====================================================
# Load All Datasets
# =====================================================

def load_all_samples():

    all_samples = []

    for dataset in DATASETS:

        print(

            f"Loading {dataset}..."

        )

        samples = load_dataset(

            dataset

        )

        print(

            f"{len(samples)} samples loaded."

        )

        all_samples.extend(

            samples

        )

    random.shuffle(

        all_samples

    )

    return all_samples