import os
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

image_features = np.load(
    os.path.join(
        BASE_DIR,
        "outputs",
        "features",
        "image_features.npy"
    )
)

text_features = np.load(
    os.path.join(
        BASE_DIR,
        "outputs",
        "captions",
        "text_embeddings.npy"
    )
)

print("Image shape:", image_features.shape)
print("Text shape :", text_features.shape)

fusion_features = np.concatenate(
    [image_features, text_features],
    axis=1
)

np.save(
    os.path.join(
        BASE_DIR,
        "outputs",
        "fusion_features.npy"
    ),
    fusion_features
)

print(
    "Fusion shape:",
    fusion_features.shape
)