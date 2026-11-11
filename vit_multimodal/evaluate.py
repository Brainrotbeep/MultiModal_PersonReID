import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

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

        for batch in tqdm(loader, leave=False):

            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

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

            features.append(feats.cpu())

            pids.extend(batch["pid"].tolist())
            cams.extend(batch["cam"].tolist())
            names.extend(batch["name"])

    features = torch.cat(features, dim=0)

    return (
        features,
        np.array(pids),
        np.array(cams),
        names
    )


# --------------------------------------------------
# Evaluate Single Query
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

    indices = np.argsort(distance_matrix[q_index])

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

    ap = np.mean(precisions)

    return cmc, ap

def save_ranking_visualization(
    distance_matrix,
    query_names,
    gallery_names,
    query_pids,
    query_cams,
    gallery_pids,
    gallery_cams,
    save_dir,
    topk=6
):

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------
    # Select Best 30 Queries
    # ------------------------------------------

    scores = []

    for q_idx in range(len(query_pids)):

        q_pid = query_pids[q_idx]
        q_cam = query_cams[q_idx]

        indices = np.argsort(distance_matrix[q_idx])

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

            indices = indices[np.logical_not(remove)[indices]]

        top_indices = indices[:topk]

        matches = (
            gallery_pids[top_indices]
            ==
            q_pid
        )
        pattern = matches.tolist()

        # Keep only examples having 1 or 2 wrong matches at the end
        if not (
                pattern == [True, True, True, True, True, False]
                or
                pattern == [True, True, True, True, False, False]
        ):
            continue

        score = matches.sum()

        scores.append((score, q_idx))

    scores.sort(reverse=True)

    selected_queries = [
        x[1]
        for x in scores[:30]
    ]

    print("\nGenerating Top-30 ranking visualizations...\n")

    # ------------------------------------------
    # Generate PNGs
    # ------------------------------------------

    for q_idx in tqdm(selected_queries):

        q_pid = query_pids[q_idx]
        q_cam = query_cams[q_idx]

        indices = np.argsort(distance_matrix[q_idx])

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

            indices = indices[np.logical_not(remove)[indices]]

        top_indices = indices[:topk]

        matches = (
            gallery_pids[top_indices]
            ==
            q_pid
        )

        fig, axes = plt.subplots(
            1,
            topk + 1,
            figsize=(14, 3)
        )

        # ------------------------------------------
        # Query Image
        # ------------------------------------------

        q_path = os.path.join(
            config.QUERY_DIR,
            query_names[q_idx]
        )

        q_img = Image.open(q_path).convert("RGB")

        axes[0].imshow(q_img)
        axes[0].set_title(
            "Query",
            fontsize=10,
            fontweight="bold"
        )
        axes[0].axis("off")

        # ------------------------------------------
        # Gallery Images
        # ------------------------------------------

        for col, g_idx in enumerate(top_indices):
            g_path = os.path.join(
                config.GALLERY_DIR,
                gallery_names[g_idx]
            )

            g_img = Image.open(g_path).convert("RGB")

            axes[col + 1].imshow(g_img)

            axes[col + 1].set_title(
                f"Rank {col + 1}",
                fontsize=10,
                fontweight="bold"
            )

            border_color = (
                "green"
                if matches[col]
                else "red"
            )

            rect = Rectangle(
                (0, 0),
                g_img.size[0] - 1,
                g_img.size[1] - 1,
                linewidth=2.5,
                edgecolor=border_color,
                facecolor="none"
            )

            axes[col + 1].add_patch(rect)

            axes[col + 1].axis("off")

        # ------------------------------------------
        # Adjust Layout
        # ------------------------------------------

        plt.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.88,
            bottom=0.05,
            wspace=0.08
        )

        # ------------------------------------------
        # Save PNG
        # ------------------------------------------

        file_name = (
                os.path.splitext(
                    query_names[q_idx]
                )[0]
                + ".png"
        )

        save_path = os.path.join(
            save_dir,
            file_name
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    print(
        f"\nSaved {len(selected_queries)} ranking visualizations to:\n{save_dir}"
    )
def evaluate(model):

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Extract Query Features
    # --------------------------------------------------

    print("\nExtracting Query Features...\n")

    query_features, query_pids, query_cams, query_names = extract_features(
        model,
        query_loader
    )

    # --------------------------------------------------
    # Extract Gallery Features
    # --------------------------------------------------

    print("\nExtracting Gallery Features...\n")

    gallery_features, gallery_pids, gallery_cams, gallery_names = extract_features(
        model,
        gallery_loader
    )

    # --------------------------------------------------
    # Compute Distance Matrix
    # --------------------------------------------------

    distance_matrix = torch.cdist(
        query_features,
        gallery_features,
        p=2
    ).cpu().numpy()

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    print("\nEvaluating...\n")

    all_cmc = []
    all_ap = []

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

    all_cmc = np.asarray(all_cmc).astype(np.float32)

    CMC = all_cmc.mean(axis=0)

    mAP = np.mean(all_ap)

    # --------------------------------------------------
    # Save Ranking Visualizations
    # --------------------------------------------------

    save_ranking_visualization(
        distance_matrix=distance_matrix,
        query_names=query_names,
        gallery_names=gallery_names,
        query_pids=query_pids,
        query_cams=query_cams,
        gallery_pids=gallery_pids,
        gallery_cams=gallery_cams,
        save_dir=os.path.join(
            config.CHECKPOINT_DIR,
            "ranking_results_mixed"
        ),
        topk=6
    )

    # --------------------------------------------------
    # Return Metrics
    # --------------------------------------------------

    return (
        float(CMC[0]),
        float(CMC[4]),
        float(CMC[9]),
        float(mAP)
    )
if __name__ == "__main__":

    print("\nLoading model...\n")

    model = MultimodalReID().to(device)

    checkpoint = os.path.join(
        config.CHECKPOINT_DIR,
        config.MODEL_NAME
    )

    print(f"Loading checkpoint:\n{checkpoint}\n")

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device
        )
    )

    model.eval()

    rank1, rank5, rank10, mAP = evaluate(model)

    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40)

    print(f"Rank-1  : {rank1:.4f}")
    print(f"Rank-5  : {rank5:.4f}")
    print(f"Rank-10 : {rank10:.4f}")
    print(f"mAP     : {mAP:.4f}")

    print("\nRanking visualizations saved in:")

    print(
        os.path.join(
            config.CHECKPOINT_DIR,
            "ranking_results_mixed"
        )
    )

    print("=" * 40)