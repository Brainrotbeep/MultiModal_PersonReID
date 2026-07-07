import torch

from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)

model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(device)