import os
import json
import numpy as np

import torch

from models.resnet50model import ReIDModel
from datasets.dataset import Market1501, transform
# paths

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

train_dir = os.path.join(
    BASE_DIR,
    "data",
    "Market-1501",
    "bounding_box_train"
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
# device

device = torch.device("cpu")
# dataset

dataset = Market1501(
    train_dir,
    transform=transform
)
# model
num_classes = len(
    set(dataset.labels)
)

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


# feature extraction

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
            f"[{idx+1}/100] "
            f"{dataset.img_paths[idx]}"
        )
# save outputs

features = np.array(features)

np.save(
    os.path.join(
        output_dir,
        "image_features.npy"
    ),
    features
)

with open(
    os.path.join(
        output_dir,
        "image_names.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        image_names,
        f,
        indent=4
    )

print("\nImage feature shape:", features.shape)
print("Saved image_features.npy")
print("Saved image_names.json")