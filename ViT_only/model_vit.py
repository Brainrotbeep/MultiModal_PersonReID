import torch
import torch.nn as nn

from transformers import ViTModel

import config_vit


class ViTBaseline(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = ViTModel.from_pretrained(
            config_vit.IMAGE_MODEL
        )

        self.dropout = nn.Dropout(
            config_vit.DROPOUT
        )

        self.classifier = nn.Linear(
            config_vit.IMAGE_FEATURE_DIM,
            config_vit.NUM_CLASSES
        )

    def forward(
        self,
        images
    ):

        outputs = self.backbone(

            pixel_values=images

        )

        features = outputs.last_hidden_state[
            :,
            0
        ]

        features = self.dropout(
            features
        )

        logits = self.classifier(
            features
        )

        return {

            "features": features,

            "logits": logits

        }