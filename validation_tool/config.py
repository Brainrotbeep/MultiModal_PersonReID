import os

# =====================================================
# Base Project Directory
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =====================================================
# Dataset Root
# =====================================================

DATASET_ROOT = r"E:\ReID_Datasets"

# =====================================================
# Caption Root
# =====================================================

CAPTION_ROOT = os.path.join(
    BASE_DIR,
    "outputs",
    "captions"
)

# =====================================================
# Output Folder
# =====================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "validation_tool",
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =====================================================
# Number of Samples
# =====================================================

SAMPLES_PER_DATASET = 50

# =====================================================
# Random Seed
# =====================================================

RANDOM_SEED = 42

# =====================================================
# Validation Attributes
# =====================================================

ATTRIBUTES = [

    "age",

    "gender",

    "upper_color",

    "upper_type",

    "lower_color",

    "lower_type",

    "footwear_color"

]

# =====================================================
# Dataset Configuration
# =====================================================

DATASETS = {

    "Market-1501": {

        "image_dir":
            os.path.join(
                DATASET_ROOT,
                "Market-1501",
                "bounding_box_test"
            ),

        "caption_file":
            os.path.join(
                CAPTION_ROOT,
                "Market-1501",
                "gallery_captions.json"
            )

    },

    "DukeMTMC-ReID": {

        "image_dir":
            os.path.join(
                DATASET_ROOT,
                "DukeMTMC-ReID",
                "bounding_box_test"
            ),

        "caption_file":
            os.path.join(
                CAPTION_ROOT,
                "DukeMTMC-ReID",
                "gallery_captions.json"
            )

    },

    "cuhk03_release": {

        "image_dir":
            os.path.join(
                DATASET_ROOT,
                "cuhk03_release",
                "detected_split",
                "gallery"
            ),

        "caption_file":
            os.path.join(
                CAPTION_ROOT,
                "cuhk03_release",
                "gallery_captions.json"
            )

    },

    "P-DukeMTMC-reid": {

        "image_dir":
            os.path.join(
                DATASET_ROOT,
                "P-DukeMTMC-reid",
                "test",
                "whole_body_images"
            ),

        "caption_file":
            os.path.join(
                CAPTION_ROOT,
                "P-DukeMTMC-reid",
                "gallery_captions.json"
            )

    },

    "Occluded_DukeMTMC": {

        "image_dir":
            os.path.join(
                DATASET_ROOT,
                "Occluded_DukeMTMC",
                "bounding_box_test"
            ),

        "caption_file":
            os.path.join(
                CAPTION_ROOT,
                "Occluded_DukeMTMC",
                "gallery_captions.json"
            )

    }

}