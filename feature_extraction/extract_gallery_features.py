import os
import json
import numpy as np

import torch

from models.resnet50model import ReIDModel
from datasets.dataset import Market1501, transform


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

gallery_dir = os.path.join(
    BASE_DIR,
    "data",
    "Market-1501",
    "bounding_box_test"
)

checkpoint_path = os.path.join(
    BASE_DIR,
    "Checkpoints",
    "resnet50",
    "final_model.pth"
)

output_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "features"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

device = torch.device("cpu")

dataset = Market1501(
    gallery_dir,
    transform=transform
)

num_classes = 751

model = ReIDModel(
    num_classes=num_classes
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.to(device)
model.eval()

features = []
image_names = []

with torch.no_grad():

    for idx in range(len(dataset)):

        img_tensor, _, _ = dataset[idx]

        img_tensor = (
            img_tensor
            .unsqueeze(0)
            .to(device)
        )

        feat, _ = model(img_tensor)

        features.append(
            feat.squeeze(0).cpu().numpy()
        )

        image_names.append(
            os.path.basename(
                dataset.img_paths[idx]
            )
        )

        print(
            f"[{idx+1}/{len(dataset)}] "
            f"{dataset.img_paths[idx]}"
        )

features = np.array(features)

np.save(
    os.path.join(
        output_dir,
        "gallery_image_features.npy"
    ),
    features
)

with open(
    os.path.join(
        output_dir,
        "gallery_image_names.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        image_names,
        f,
        indent=4
    )

print(
    "\nGallery feature shape:",
    features.shape
)