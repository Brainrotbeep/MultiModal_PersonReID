import os
import torch

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    BASE_DIR
)

# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# --------------------------------------------------
# Dataset Selection
# --------------------------------------------------

DATASET_ROOT = r"E:\ReID_Datasets"

DATASET = "Occluded_DukeMTMC"

# Available datasets
# "Market-1501"
# "DukeMTMC-ReID"
# "Occluded_DukeMTMC"
# "cuhk03_release"
# "Occluded_REID"
# "P-DukeMTMC-reid"

# --------------------------------------------------
# Dataset Paths
# --------------------------------------------------

if DATASET in [
    "Market-1501",
    "DukeMTMC-ReID",
    "Occluded_DukeMTMC"
]:

    TRAIN_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "bounding_box_train"
    )

    QUERY_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "query"
    )

    GALLERY_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "bounding_box_test"
    )


elif DATASET == "cuhk03_release":

    DATASET_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "detected_split"
    )

    TRAIN_DIR = os.path.join(
        DATASET_DIR,
        "train"
    )

    QUERY_DIR = os.path.join(
        DATASET_DIR,
        "query"
    )

    GALLERY_DIR = os.path.join(
        DATASET_DIR,
        "gallery"
    )


elif DATASET == "Occluded_REID":

    TRAIN_DIR = None

    QUERY_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "occluded_body_images"
    )

    GALLERY_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "whole_body_images"
    )


elif DATASET == "P-DukeMTMC-reid":

    TRAIN_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "train"
    )

    QUERY_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "test",
        "occluded_body_images"
    )

    GALLERY_DIR = os.path.join(
        DATASET_ROOT,
        DATASET,
        "test",
        "whole_body_images"
    )

else:

    raise ValueError(
        f"Unsupported dataset: {DATASET}"
    )


# --------------------------------------------------
# Dataset Type
# --------------------------------------------------

IS_OCCLUDED = DATASET in [

    "Occluded_REID",

    "P-DukeMTMC-reid"

]
# --------------------------------------------------
# Image Extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = (

    ".jpg",

    ".jpeg",

    ".png",

    ".tif",

    ".tiff"

)

# --------------------------------------------------
# Number of Classes
# --------------------------------------------------

NUM_CLASSES = {

    "Market-1501": 751,

    "DukeMTMC-ReID": 702,

    "Occluded_DukeMTMC": 702,

    "cuhk03_release": 767,

    "Occluded_REID": 200,

    "P-DukeMTMC-reid": 665

}[DATASET]

# --------------------------------------------------
# Image Encoder
# --------------------------------------------------

IMAGE_MODEL = "google/vit-base-patch16-224"

IMAGE_SIZE = 224

IMAGE_FEATURE_DIM = 768

DROPOUT = 0.3

# --------------------------------------------------
# Training
# --------------------------------------------------

EPOCHS = 20

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 1e-4

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

TOP_K = [1, 5, 10]

# --------------------------------------------------
# ViT Baseline Checkpoints
# --------------------------------------------------

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints",
    DATASET
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

MODEL_NAME = f"vit_baseline_{DATASET}.pth"

BEST_MODEL_NAME = f"vit_baseline_{DATASET}_best.pth"
# --------------------------------------------------
# Dataset Switcher
# --------------------------------------------------

def set_dataset(dataset_name):

    global DATASET
    global TRAIN_DIR
    global QUERY_DIR
    global GALLERY_DIR
    global NUM_CLASSES
    global CHECKPOINT_DIR
    global MODEL_NAME
    global BEST_MODEL_NAME
    global IS_OCCLUDED

    DATASET = dataset_name

    IS_OCCLUDED = DATASET in [

        "Occluded_REID",

        "P-DukeMTMC-reid"

    ]

    if DATASET in [

        "Market-1501",

        "DukeMTMC-ReID",

        "Occluded_DukeMTMC"

    ]:

        TRAIN_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "bounding_box_train"
        )

        QUERY_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "query"
        )

        GALLERY_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "bounding_box_test"
        )

    elif DATASET == "cuhk03_release":

        DATASET_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "detected_split"
        )

        TRAIN_DIR = os.path.join(
            DATASET_DIR,
            "train"
        )

        QUERY_DIR = os.path.join(
            DATASET_DIR,
            "query"
        )

        GALLERY_DIR = os.path.join(
            DATASET_DIR,
            "gallery"
        )

    elif DATASET == "Occluded_REID":

        TRAIN_DIR = None

        QUERY_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "occluded_body_images"
        )

        GALLERY_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "whole_body_images"
        )

    elif DATASET == "P-DukeMTMC-reid":

        TRAIN_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "train"
        )

        QUERY_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "test",
            "occluded_body_images"
        )

        GALLERY_DIR = os.path.join(
            DATASET_ROOT,
            DATASET,
            "test",
            "whole_body_images"
        )

    else:

        raise ValueError(
            f"Unsupported dataset: {DATASET}"
        )

    NUM_CLASSES = {

        "Market-1501": 751,

        "DukeMTMC-ReID": 702,

        "Occluded_DukeMTMC": 702,

        "cuhk03_release": 767,

        "Occluded_REID": 200,

        "P-DukeMTMC-reid": 665

    }[DATASET]

    CHECKPOINT_DIR = os.path.join(
        BASE_DIR,
        "checkpoints",
        DATASET
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    MODEL_NAME = f"vit_baseline_{DATASET}.pth"

    BEST_MODEL_NAME = (
        f"vit_baseline_{DATASET}_best.pth"
    )
