import os
import json
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

query_features = np.load(
    os.path.join(
        BASE_DIR,
        "outputs",
        "fusion",
        "query_fusion_features.npy"
    )
)

gallery_features = np.load(
    os.path.join(
        BASE_DIR,
        "outputs",
        "fusion",
        "gallery_fusion_features.npy"
    )
)

with open(
    os.path.join(
        BASE_DIR,
        "outputs",
        "features",
        "query_image_names.json"
    ),
    "r"
) as f:

    query_names = json.load(f)

with open(
    os.path.join(
        BASE_DIR,
        "outputs",
        "features",
        "gallery_image_names.json"
    ),
    "r"
) as f:

    gallery_names = json.load(f)


def get_pid_cam(filename):

    parts = filename.split("_")

    pid = int(parts[0])

    cam = int(
        parts[1][1]
    )

    return pid, cam


query_pids = []
query_cams = []

for name in query_names:

    pid, cam = get_pid_cam(name)

    query_pids.append(pid)
    query_cams.append(cam)

gallery_pids = []
gallery_cams = []

for name in gallery_names:

    pid, cam = get_pid_cam(name)

    gallery_pids.append(pid)
    gallery_cams.append(cam)


CMC = np.zeros(
    len(gallery_names)
)

APs = []

for i in range(
    len(query_features)
):

    q_feat = query_features[i]

    q_pid = query_pids[i]

    q_cam = query_cams[i]

    distances = np.linalg.norm(
        gallery_features - q_feat,
        axis=1
    )

    order = np.argsort(
        distances
    )

    remove = []

    keep = []

    for idx in order:

        if (
            gallery_pids[idx] == q_pid
            and
            gallery_cams[idx] == q_cam
        ):
            remove.append(idx)

        else:
            keep.append(idx)

    matches = (
        np.array(
            gallery_pids
        )[keep]
        ==
        q_pid
    ).astype(int)

    if matches.sum() == 0:
        continue

    first_match = np.where(
        matches == 1
    )[0][0]

    CMC[
        first_match:
    ] += 1

    precisions = []

    correct = 0

    for rank, match in enumerate(matches):

        if match:

            correct += 1

            precisions.append(
                correct / (rank + 1)
            )

    AP = np.mean(
        precisions
    )

    APs.append(AP)

CMC = (
    CMC /
    len(APs)
)

mAP = np.mean(
    APs
)

print("\n RESULTS ")

print(
    f"Rank-1 : {CMC[0]:.4f}"
)

print(
    f"Rank-5 : {CMC[4]:.4f}"
)

print(
    f"Rank-10: {CMC[9]:.4f}"
)

print(
    f"mAP    : {mAP:.4f}"
)