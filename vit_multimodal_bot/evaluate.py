import os
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

import config

if config.DATASET in [
    "Occluded_REID",
    "P-DukeMTMC-reid"
]:
    from dataset_occluded import PersonReIDDataset
else:
    from dataset import PersonReIDDataset

from model import MultimodalReID


device = torch.device(config.DEVICE)


def extract_features(model, loader):

    model.eval()

    features = []
    pids = []
    cams = []
    names = []

    with torch.no_grad():

        for batch in tqdm(loader, leave=False):

            images = batch["image"].to(device)

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            outputs = model(

                images,

                input_ids,

                attention_mask

            )

            # --------------------------------------------------
            # Use BNNeck features for retrieval
            # --------------------------------------------------

            feats = F.normalize(

                outputs["bn_features"],

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

    indices = np.argsort(
        distance_matrix[q_index]
    )

    if config.DATASET in [
        "Market-1501",
        "DukeMTMC-ReID",
        "Occluded_DukeMTMC",
        "cuhk03_release"
    ]:

        remove = (
                (gallery_pids == q_pid)
                &
                (gallery_cams == q_cam)
        )

    else:

        remove = np.zeros_like(
            gallery_pids,
            dtype=bool
        )

    indices = indices[
        np.logical_not(remove)[indices]
    ]

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


def evaluate(model):

    query_dataset = PersonReIDDataset(
        image_dir=config.QUERY_DIR,
        caption_file=config.QUERY_CAPTIONS,
        train=False
    )

    gallery_dataset = PersonReIDDataset(
        image_dir=config.GALLERY_DIR,
        caption_file=config.GALLERY_CAPTIONS,
        train=False
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.BATCH_SIZE,
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

    distance_matrix = torch.cdist(
        query_features,
        gallery_features,
        p=2
    ).numpy()

    all_cmc = []
    all_ap = []

    print("\nEvaluating...\n")

    for q_idx in tqdm(
        range(len(query_pids))
    ):

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


if __name__ == "__main__":

    model = MultimodalReID().to(
        device
    )

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

    rank1, rank5, rank10, mAP = evaluate(
        model
    )

    print("\nEvaluation Results\n")

    print(
        f"Rank-1  : {rank1:.4f}"
    )

    print(
        f"Rank-5  : {rank5:.4f}"
    )

    print(
        f"Rank-10 : {rank10:.4f}"
    )

    print(
        f"mAP     : {mAP:.4f}"
    )