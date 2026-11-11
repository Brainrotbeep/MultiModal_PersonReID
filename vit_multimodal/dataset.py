import os
import json

from PIL import Image

from torch.utils.data import Dataset

from transformers import ViTImageProcessor
from transformers import DistilBertTokenizer

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

        for img_name in sorted(os.listdir(image_dir)):

            if not (
                img_name.endswith(".jpg")
                or
                img_name.endswith(".png")
            ):
                continue

            if img_name not in self.captions:
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

                "caption": self.captions[
                    img_name
                ],

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

        if config.DATASET == "cuhk03_release":

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

    def make_sentence(
        self,
        attrs
    ):

        return (

            f"Age: {attrs['age']}. "

            f"Gender: {attrs['gender']}. "

            f"Upper clothing: {attrs['upper_color']} {attrs['upper_type']}. "

            f"Lower clothing: {attrs['lower_color']} {attrs['lower_type']}. "

            f"Footwear Color: {attrs['footwear_color']}."

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

            "input_ids": tokens["input_ids"].squeeze(0),

            "attention_mask": tokens["attention_mask"].squeeze(0),

            "label": label,

            "pid": sample["pid"],

            "cam": sample["cam"],

            "name": sample["name"]

        }