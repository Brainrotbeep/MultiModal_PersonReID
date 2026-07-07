import os
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

image_features = np.load(
    os.path.join(
        BASE_DIR,
        "outputs",
        "features",
        "gallery_image_features.npy"
    )
)

text_features = np.load(
    os.path.join(
        BASE_DIR,
        "outputs",
        "text_embeddings",
        "gallery_embeddings.npy"
    )
)

print("Gallery Image shape:", image_features.shape)
print("Gallery Text shape :", text_features.shape)

fusion_features = np.concatenate(
    [image_features, text_features],
    axis=1
)

output_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "fusion"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

np.save(
    os.path.join(
        output_dir,
        "gallery_fusion_features.npy"
    ),
    fusion_features
)

print(
    "Gallery Fusion shape:",
    fusion_features.shape
)