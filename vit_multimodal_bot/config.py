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

DATASET = "Market-1501"

# --------------------------------------------------
# Dataset Paths
# --------------------------------------------------

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

IS_OCCLUDED = False

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

NUM_CLASSES = 751

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

# PK Sampler
P = 8
K = 2
BATCH_SIZE = P * K

# Optimizer
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-4

# Bag of Tricks
USE_BNNECK = True
USE_TRIPLET = True
USE_CENTER_LOSS = False

TRIPLET_MARGIN = 0.3

ID_LOSS_WEIGHT = 1.0
TRIPLET_LOSS_WEIGHT = 1.0
CENTER_LOSS_WEIGHT = 5e-4

# Label Smoothing
LABEL_SMOOTHING = 0.1

# Gradient Clipping
GRAD_CLIP = 1.0

# Learning Rate Scheduler
USE_WARMUP = True
WARMUP_EPOCHS = 10

USE_COSINE = True
MIN_LR = 1e-7

# Data Augmentation
USE_RANDOM_ERASING = True
ERASING_PROB = 0.5

# -------------------------
# Evaluation
# -------------------------

TOP_K = [1, 5, 10]

# Evaluate model after every N epochs
EVAL_INTERVAL = 10

# --------------------------------------------------
# Checkpoints
# --------------------------------------------------
# Save checkpoint after every N epochs
SAVE_INTERVAL = 10

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "vit_multimodal_bot",
    "checkpoints",
    DATASET
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

MODEL_NAME = f"vit_multimodal_bot_{DATASET}.pth"