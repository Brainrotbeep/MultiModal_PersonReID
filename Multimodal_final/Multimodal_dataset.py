import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):

    def __init__(self):

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.features = np.load(
            os.path.join(
                base_dir,
                "outputs",
                "fusion_features.npy"
            )
        )

        with open(
            os.path.join(
                base_dir,
                "outputs",
                "features",
                "image_names.json"
            ),
            "r"
        ) as f:

            self.image_names = json.load(f)

        self.labels = []

        pids = []

        for img in self.image_names:

            pid = int(
                img.split("_")[0]
            )

            pids.append(pid)

        pid_map = {
            pid: idx
            for idx, pid in enumerate(
                sorted(set(pids))
            )
        }

        for pid in pids:

            self.labels.append(
                pid_map[pid]
            )

    def __len__(self):

        return len(
            self.features
        )

    def __getitem__(self, idx):

        feature = torch.tensor(
            self.features[idx],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return feature, label