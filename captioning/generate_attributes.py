from PIL import Image

import torch

from captioning.blip_model import (
    processor,
    model,
    device
)

from captioning.prompts import PROMPTS


model.eval()


def get_attributes_batch(image_paths):

    images = []

    valid_names = []

    for path in image_paths:

        try:

            image = Image.open(
                path
            ).convert("RGB")

            images.append(image)

            valid_names.append(path)

        except Exception:

            print(f"Unable to open: {path}")

    results = {

        path: {}

        for path in valid_names

    }

    with torch.no_grad():

        for key, question in PROMPTS.items():

            questions = [

                question

                for _ in images

            ]

            inputs = processor(

                images=images,

                text=questions,

                return_tensors="pt",

                padding=True

            ).to(device)

            outputs = model.generate(

                **inputs,

                max_new_tokens=20

            )

            answers = processor.batch_decode(

                outputs,

                skip_special_tokens=True

            )

            for path, answer in zip(

                valid_names,

                answers

            ):

                answer = answer.strip().lower()

                if answer == "":

                    answer = "unknown"

                results[path][key] = answer

    return results