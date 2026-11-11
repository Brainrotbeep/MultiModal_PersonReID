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

        for img_name in sorted(os.listdir(image_dir)):

            if not img_name.lower().endswith(
                config_vit.IMAGE_EXTENSIONS
            ):
                continue

            pid, cam = self.parse_filename(
                img_name
            )

            if pid == -1:
                continue

            self.samples.append({

                "image": os.path.join(
                    image_dir,
                    img_name
                ),

                "pid": pid,

                "cam": cam,

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

    def parse_filename(
        self,
        img_name
    ):

        if config_vit.DATASET == "cuhk03_release":

            pid = int(
                img_name.split("_")[0]
            )

            cam = int(
                img_name.split("_")[1]
            )

        else:

            pid = int(
                img_name.split("_")[0]
            )

            cam = int(
                img_name.split("_")[1][1]
            )

        return pid, cam

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