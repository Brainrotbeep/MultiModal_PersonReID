import os
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

import config

from dataset import Market1501Dataset
from model import MultimodalReID


device = torch.device(config.DEVICE)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

model = MultimodalReID().to(device)

checkpoint = os.path.join(
    config.CHECKPOINT_DIR,
    config.MODEL_NAME
)

model.load_state_dict(
    torch.load(
        checkpoint,
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# Query Dataset
# --------------------------------------------------

query_dataset = Market1501Dataset(
    image_dir=config.QUERY_DIR,
    caption_file=config.QUERY_CAPTIONS,
    train=False
)

query_loader = DataLoader(
    query_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Gallery Dataset
# --------------------------------------------------

gallery_dataset = Market1501Dataset(
    image_dir=config.GALLERY_DIR,
    caption_file=config.GALLERY_CAPTIONS,
    train=False
)

gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Feature Extraction
# --------------------------------------------------

def extract_features(loader):

    features = []

    pids = []

    cams = []

    names = []

    with torch.no_grad():

        for batch in tqdm(loader):

            images = batch["image"].to(device)

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            outputs = model(

                images,

                input_ids,

                attention_mask

            )

            feats = outputs["fusion_features"]

            # L2 Normalization
            feats = F.normalize(
                feats,
                p=2,
                dim=1
            )

            features.append(
                feats.cpu()
            )

            pids.extend(
                batch["pid"].tolist()
            )

            cams.extend(
                batch["cam"].tolist()
            )

            names.extend(
                batch["name"]
            )

    features = torch.cat(
        features,
        dim=0
    )

    return (

        features,

        np.array(pids),

        np.array(cams),

        names

    )


print("\nExtracting Query Features...\n")

query_features, query_pids, query_cams, query_names = extract_features(
    query_loader
)


print("\nExtracting Gallery Features...\n")

gallery_features, gallery_pids, gallery_cams, gallery_names = extract_features(
    gallery_loader
)


print("\nFeature Extraction Completed\n")

print(
    "Query Features :",
    query_features.shape
)

print(
    "Gallery Features :",
    gallery_features.shape
)
# --------------------------------------------------
# Distance Matrix
# --------------------------------------------------

distance_matrix = torch.cdist(

    query_features,

    gallery_features,

    p=2

).numpy()

print("\nDistance Matrix Shape :",
      distance_matrix.shape)


# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate_query(

    q_index,

    distance_matrix,

    query_pids,

    query_cams,

    gallery_pids,

    gallery_cams

):

    q_pid = query_pids[q_index]

    q_cam = query_cams[q_index]

    distances = distance_matrix[q_index]

    indices = np.argsort(
        distances
    )

    # Remove same person from same camera

    remove = (

        (gallery_pids == q_pid)

        &

        (gallery_cams == q_cam)

    )

    keep = np.logical_not(
        remove
    )

    indices = indices[
        keep[indices]
    ]

    matches = (

        gallery_pids[indices]

        ==

        q_pid

    ).astype(np.int32)

    if matches.sum() == 0:

        return None, None

    # ----------------------------
    # CMC
    # ----------------------------

    first_match = np.where(
        matches == 1
    )[0][0]

    cmc = np.zeros(
        len(gallery_pids)
    )

    cmc[first_match:] = 1

    # ----------------------------
    # Average Precision
    # ----------------------------

    precisions = []

    correct = 0

    for rank, match in enumerate(matches):

        if match:

            correct += 1

            precisions.append(

                correct /

                (rank + 1)

            )

    ap = np.mean(
        precisions
    )

    return cmc, ap


# --------------------------------------------------
# Containers
# --------------------------------------------------

all_cmc = []

all_ap = []
# --------------------------------------------------
# Evaluate All Queries
# --------------------------------------------------

print("\nEvaluating...\n")

for q_idx in tqdm(range(len(query_pids))):

    cmc, ap = evaluate_query(

        q_idx,

        distance_matrix,

        query_pids,

        query_cams,

        gallery_pids,

        gallery_cams

    )

    if cmc is None:
        continue

    all_cmc.append(cmc)

    all_ap.append(ap)


# --------------------------------------------------
# Final Metrics
# --------------------------------------------------

all_cmc = np.asarray(all_cmc).astype(np.float32)

CMC = all_cmc.mean(axis=0)

mAP = np.mean(all_ap)


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print(" Market1501 Evaluation Results")

print(f"Rank-1  : {CMC[0]:.4f}")
print(f"Rank-5  : {CMC[4]:.4f}")
print(f"Rank-10 : {CMC[9]:.4f}")
print(f"mAP     : {mAP:.4f}")
