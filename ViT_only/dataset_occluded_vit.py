import os

from PIL import Image

from torch.utils.data import Dataset

from transformers import ViTImageProcessor

import config_vit


class PersonReIDDataset(Dataset):

    def __init__(
        self,
        image_dir,
        train=True
    ):

        self.image_dir = image_dir
        self.train = train

        self.processor = ViTImageProcessor.from_pretrained(
            config_vit.IMAGE_MODEL
        )

        self.samples = []

        pids = set()

        for root, _, files in os.walk(image_dir):

            pid_folder = os.path.basename(root)

            if not pid_folder.isdigit():
                continue

            pid = int(pid_folder)

            for img_name in sorted(files):

                if not img_name.lower().endswith(
                    config_vit.IMAGE_EXTENSIONS
                ):
                    continue

                full_path = os.path.join(
                    root,
                    img_name
                )

                self.samples.append({

                    "image": full_path,

                    "pid": pid,

                    "cam": 0,

                    "name": img_name

                })

                pids.add(pid)

        if train:

            self.pid2label = {

                pid: idx

                for idx, pid in enumerate(
                    sorted(pids)
                )

            }

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]

        image = Image.open(
            sample["image"]
        ).convert("RGB")

        image = self.processor(

            images=image,

            return_tensors="pt"

        )["pixel_values"].squeeze(0)

        if self.train:

            label = self.pid2label[
                sample["pid"]
            ]

        else:

            label = sample["pid"]

        return {

            "image": image,

            "label": label,

            "pid": sample["pid"],

            "cam": sample["cam"],

            "name": sample["name"]

        }