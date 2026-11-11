import os
import torch

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

DATASET ="Market-1501"

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
# Caption Files
# --------------------------------------------------

CAPTION_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "captions",
    DATASET
)

os.makedirs(
    CAPTION_DIR,
    exist_ok=True
)

TRAIN_CAPTIONS = os.path.join(
    CAPTION_DIR,
    "train_captions.json"
)

QUERY_CAPTIONS = os.path.join(
    CAPTION_DIR,
    "query_captions.json"
)

GALLERY_CAPTIONS = os.path.join(
    CAPTION_DIR,
    "gallery_captions.json"
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

# --------------------------------------------------
# Text Encoder
# --------------------------------------------------

TEXT_MODEL = "distilbert-base-uncased"

MAX_TEXT_LENGTH = 64

# --------------------------------------------------
# Fusion
# --------------------------------------------------

IMAGE_FEATURE_DIM = 768

TEXT_FEATURE_DIM = 768

FUSION_DIM = 768

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
# Checkpoints
# --------------------------------------------------

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "vit_multimodal",
    "checkpoints",
    DATASET
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

MODEL_NAME = f"vit_multimodal_{DATASET}.pth"
