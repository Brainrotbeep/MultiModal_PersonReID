import torch
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
# Device

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Dataset

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "Market-1501",
    "bounding_box_train"
)

QUERY_DIR = os.path.join(
    BASE_DIR,
    "data",
    "Market-1501",
    "query"
)

GALLERY_DIR = os.path.join(
    BASE_DIR,
    "data",
    "Market-1501",
    "bounding_box_test"
)

TRAIN_CAPTIONS = os.path.join(
    BASE_DIR,
    "outputs",
    "captions",
    "train_captions.json"
)

QUERY_CAPTIONS = os.path.join(
    BASE_DIR,
    "outputs",
    "captions",
    "query_captions.json"
)

GALLERY_CAPTIONS = os.path.join(
    BASE_DIR,
    "outputs",
    "captions",
    "gallery_captions.json"
)
NUM_CLASSES = 751

# Image Encoder

IMAGE_MODEL = "google/vit-base-patch16-224"

IMAGE_SIZE = 224

# Text Encoder

TEXT_MODEL = "distilbert-base-uncased"

MAX_TEXT_LENGTH = 64

# Fusion

IMAGE_FEATURE_DIM = 768
TEXT_FEATURE_DIM = 768

FUSION_DIM = 768

DROPOUT = 0.3

# Training

EPOCHS = 5

BATCH_SIZE = 16

LEARNING_RATE = 3e-5

WEIGHT_DECAY = 1e-4

# Evaluation


TOP_K = [1, 5, 10]
# Paths

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "vit_multimodal",
    "checkpoints"
)

MODEL_NAME = "vit_multimodal.pth"