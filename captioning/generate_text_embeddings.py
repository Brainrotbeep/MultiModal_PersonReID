import os
import json
import numpy as np

from sentence_transformers import SentenceTransformer

# Project root
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

caption_file = os.path.join(
    BASE_DIR,
    "outputs",
    "captions",
    "train_captions.json"
)

embedding_file = os.path.join(
    BASE_DIR,
    "outputs",
    "captions",
    "text_embeddings.npy"
)

image_names_file = os.path.join(
    BASE_DIR,
    "outputs",
    "captions",
    "image_names.json"
)

print("Caption file:", caption_file)
print("Exists:", os.path.exists(caption_file))

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Load captions
with open(
    caption_file,
    "r",
    encoding="utf-8"
) as f:
    captions = json.load(f)

image_names = []
texts = []

for img, attrs in captions.items():

    text = " ".join([
        attrs["age"],
        attrs["gender"],
        attrs["upper_color"],
        attrs["upper_type"],
        attrs["lower_color"],
        attrs["lower_type"],
        attrs["footwear"]
    ])

    image_names.append(img)
    texts.append(text)

# Generate embeddings
embeddings = model.encode(
    texts,
    show_progress_bar=True
)

# Save embeddings
np.save(
    embedding_file,
    embeddings
)

# Save image names
with open(
    image_names_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        image_names,
        f,
        indent=4
    )

print("Embeddings shape:", embeddings.shape)
print("Saved:", embedding_file)
print("Saved:", image_names_file)