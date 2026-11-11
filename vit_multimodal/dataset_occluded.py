import os
import json

from PIL import Image

from torch.utils.data import Dataset

from transformers import (
    ViTImageProcessor,
    DistilBertTokenizer
)

import config


class PersonReIDDataset(Dataset):

    def __init__(
        self,
        image_dir,
        caption_file,
        train=True
    ):

        self.image_dir = image_dir
        self.train = train

        with open(
            caption_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.captions = json.load(f)

        self.processor = ViTImageProcessor.from_pretrained(
            config.IMAGE_MODEL
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            config.TEXT_MODEL
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
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".tif",
                        ".tiff"
                    )
                ):
                    continue

                full_path = os.path.join(
                    root,
                    img_name
                )

                # ------------------------------
                # Caption Key
                # ------------------------------

                if config.DATASET in [

                    "P-DukeMTMC-reid",

                    "Occluded_REID"

                ]:

                    caption_key = os.path.relpath(
                        full_path,
                        image_dir
                    ).replace("\\", "/")

                else:

                    caption_key = img_name

                if caption_key not in self.captions:
                    continue

                self.samples.append({

                    "image": full_path,

                    "caption": self.captions[
                        caption_key
                    ],

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

    def make_sentence(
            self,
            attrs
    ):

        return (

            f"Age: {attrs.get('age', 'unknown')}. "

            f"Gender: {attrs.get('gender', 'unknown')}. "

            f"Upper clothing: "
            f"{attrs.get('upper_color', 'unknown')} "
            f"{attrs.get('upper_type', 'unknown')}. "

            f"Lower clothing: "
            f"{attrs.get('lower_color', 'unknown')} "
            f"{attrs.get('lower_type', 'unknown')}. "

            f"Footwear: "
            f"{attrs.get('footwear_color', 'unknown')}."

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

        caption = self.make_sentence(
            sample["caption"]
        )

        tokens = self.tokenizer(

            caption,

            padding="max_length",

            truncation=True,

            max_length=config.MAX_TEXT_LENGTH,

            return_tensors="pt"

        )

        if self.train:

            label = self.pid2label[
                sample["pid"]
            ]

        else:

            label = sample["pid"]

        return {

            "image": image,

            "input_ids": tokens[
                "input_ids"
            ].squeeze(0),

            "attention_mask": tokens[
                "attention_mask"
            ].squeeze(0),

            "label": label,

            "pid": sample["pid"],

            "cam": sample["cam"],

            "name": sample["name"]

        }