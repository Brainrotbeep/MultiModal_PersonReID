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
        "query_image_features.npy"
    )
)

text_features = np.load(
    os.path.join(
        BASE_DIR,
        "outputs",
        "text_embeddings",
        "query_embeddings.npy"
    )
)

print("Query Image shape:", image_features.shape)
print("Query Text shape :", text_features.shape)

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
        "query_fusion_features.npy"
    ),
    fusion_features
)

print(
    "Query Fusion shape:",
    fusion_features.shape
)