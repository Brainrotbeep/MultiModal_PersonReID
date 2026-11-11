import os
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

import config

from dataset_occluded import PersonReIDDataset
from model import MultimodalReID


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(config.DEVICE)


# --------------------------------------------------
# Cross Dataset Configuration
# --------------------------------------------------

# Occluded_REID Dataset

QUERY_DIR = (
    r"E:\ReID_Datasets\Occluded_REID"
    r"\occluded_body_images"
)

GALLERY_DIR = (
    r"E:\ReID_Datasets\Occluded_REID"
    r"\whole_body_images"
)

# Caption Files

QUERY_CAPTIONS = (
    r"E:\Gargi\MultiModal_PersonReID"
    r"\outputs\captions"
    r"\Occluded_REID"
    r"\query_captions.json"
)

GALLERY_CAPTIONS = (
    r"E:\Gargi\MultiModal_PersonReID"
    r"\outputs\captions"
    r"\Occluded_REID"
    r"\gallery_captions.json"
)

# Trained Model (Occluded Duke)

CHECKPOINT = (
    r"E:\Gargi\MultiModal_PersonReID"
    r"\vit_multimodal\checkpoints"
    r"\best_epoch20_od.pth"
)

BATCH_SIZE = config.BATCH_SIZE


# --------------------------------------------------
# Feature Extraction
# --------------------------------------------------

def extract_features(model, loader):

    model.eval()

    features = []
    pids = []
    cams = []
    names = []

    with torch.no_grad():

        for batch in tqdm(
            loader,
            leave=False,
            desc="Extracting"
        ):

            images = batch["image"].to(device)

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            outputs = model(
                images,
                input_ids,
                attention_mask
            )

            feats = F.normalize(
                outputs["fusion_features"],
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
# --------------------------------------------------
# Single Query Evaluation
# --------------------------------------------------

def evaluate_query(

    q_index,

    distance_matrix,

    query_pids,

    gallery_pids

):

    q_pid = query_pids[q_index]

    indices = np.argsort(

        distance_matrix[q_index]

    )

    matches = (

        gallery_pids[indices]

        ==

        q_pid

    ).astype(np.int32)

    if matches.sum() == 0:

        return None, None

    first_match = np.where(

        matches == 1

    )[0][0]

    cmc = np.zeros(

        len(gallery_pids)

    )

    cmc[first_match:] = 1

    precisions = []

    correct = 0

    for rank, match in enumerate(matches):

        if match:

            correct += 1

            precisions.append(

                correct / (rank + 1)

            )

    ap = np.mean(

        precisions

    )

    return cmc, ap


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model):

    print("\n========================================")
    print("Cross Dataset Evaluation")
    print("========================================")
    print("Training Dataset : Occluded_DukeMTMC")
    print("Testing Dataset  : Occluded_REID")
    print("========================================\n")

    query_dataset = PersonReIDDataset(

        image_dir=QUERY_DIR,

        caption_file=QUERY_CAPTIONS,

        train=False

    )

    gallery_dataset = PersonReIDDataset(

        image_dir=GALLERY_DIR,

        caption_file=GALLERY_CAPTIONS,

        train=False

    )
    print("\n========================================")
    print("Dataset Debug")
    print("========================================")

    print(f"Query Caption File   : {config.QUERY_CAPTIONS}")
    print(f"Gallery Caption File : {config.GALLERY_CAPTIONS}")

    print(f"Query Samples        : {len(query_dataset)}")
    print(f"Gallery Samples      : {len(gallery_dataset)}")
    print("========================================\n")

    query_loader = DataLoader(

        query_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )

    gallery_loader = DataLoader(

        gallery_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )

    print("\nExtracting Query Features...\n")

    query_features, query_pids, query_cams, _ = extract_features(

        model,

        query_loader

    )

    print("\nExtracting Gallery Features...\n")

    gallery_features, gallery_pids, gallery_cams, _ = extract_features(

        model,

        gallery_loader

    )

    print("\nComputing Distance Matrix...\n")

    distance_matrix = torch.cdist(

        query_features,

        gallery_features,

        p=2

    ).cpu().numpy()

    all_cmc = []

    all_ap = []

    print("\nEvaluating...\n")

    for q_idx in tqdm(

        range(len(query_pids)),

        desc="Query"

    ):

        cmc, ap = evaluate_query(

            q_idx,

            distance_matrix,

            query_pids,

            gallery_pids

        )

        if cmc is None:

            continue

        all_cmc.append(cmc)

        all_ap.append(ap)

    all_cmc = np.asarray(

        all_cmc

    ).astype(np.float32)

    CMC = all_cmc.mean(

        axis=0

    )

    mAP = np.mean(

        all_ap

    )

    return (

        float(CMC[0]),

        float(CMC[4]),

        float(CMC[9]),

        float(mAP)

    )
# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("\n========================================")
    print("Loading Cross-Dataset Model")
    print("========================================")

    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Query Dir  : {QUERY_DIR}")
    print(f"Gallery Dir: {GALLERY_DIR}")

    print("========================================\n")

    if not os.path.exists(CHECKPOINT):

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    if not os.path.exists(QUERY_CAPTIONS):

        raise FileNotFoundError(
            f"Query caption file not found:\n{QUERY_CAPTIONS}"
        )

    if not os.path.exists(GALLERY_CAPTIONS):

        raise FileNotFoundError(
            f"Gallery caption file not found:\n{GALLERY_CAPTIONS}"
        )

    model = MultimodalReID().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    # Remove classifier weights
    checkpoint.pop("classifier.weight", None)
    checkpoint.pop("classifier.bias", None)

    model.load_state_dict(
        checkpoint,
        strict=False
    )

    print("Model Loaded Successfully.\n")

    rank1, rank5, rank10, mAP = evaluate(
        model
    )

    print("\n========================================")
    print("Cross-Dataset Evaluation Results")
    print("========================================")

    print(f"Rank-1  : {rank1:.4f}")
    print(f"Rank-5  : {rank5:.4f}")
    print(f"Rank-10 : {rank10:.4f}")
    print(f"mAP     : {mAP:.4f}")
