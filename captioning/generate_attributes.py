from PIL import Image

from captioning.blip_model import (
    processor,
    model,
    device
)

from captioning.prompts import PROMPTS


def get_attributes(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    attributes = {}

    for key, question in PROMPTS.items():

        inputs = processor(
            image,
            question,
            return_tensors="pt"
        ).to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=20
        )

        answer = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        attributes[key] = answer

    return attributes