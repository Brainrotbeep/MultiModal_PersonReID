import torch
import torch.nn as nn

from transformers import (
    ViTModel,
    DistilBertModel
)

import config


class MultimodalReID(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Image Encoder
        # --------------------------------------------------

        self.image_encoder = ViTModel.from_pretrained(
            config.IMAGE_MODEL
        )

        # --------------------------------------------------
        # Text Encoder
        # --------------------------------------------------

        self.text_encoder = DistilBertModel.from_pretrained(
            config.TEXT_MODEL
        )

        # --------------------------------------------------
        # Fusion Layer
        # --------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                config.IMAGE_FEATURE_DIM +
                config.TEXT_FEATURE_DIM,
                config.FUSION_DIM
            ),

            nn.BatchNorm1d(
                config.FUSION_DIM
            ),

            nn.ReLU(),

            nn.Dropout(
                config.DROPOUT
            )

        )

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        self.classifier = nn.Linear(

            config.FUSION_DIM,

            config.NUM_CLASSES

        )

        # --------------------------------------------------
        # Weight Initialization
        # --------------------------------------------------

        nn.init.xavier_uniform_(
            self.fusion[0].weight
        )

        nn.init.zeros_(
            self.fusion[0].bias
        )

        nn.init.xavier_uniform_(
            self.classifier.weight
        )

        nn.init.zeros_(
            self.classifier.bias
        )

    def forward(

        self,

        images,

        input_ids,

        attention_mask

    ):

        # --------------------------------------------------
        # Image Features (ViT CLS Token)
        # --------------------------------------------------

        image_outputs = self.image_encoder(
            pixel_values=images
        )

        image_features = image_outputs.last_hidden_state[
            :, 0
        ]

        # --------------------------------------------------
        # Text Features (DistilBERT CLS Token)
        # --------------------------------------------------

        text_outputs = self.text_encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        text_features = text_outputs.last_hidden_state[
            :, 0
        ]

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        fused_features = torch.cat(

            [

                image_features,

                text_features

            ],

            dim=1

        )

        fused_features = self.fusion(
            fused_features
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.classifier(
            fused_features
        )

        return {

            "logits": logits,

            "image_features": image_features,

            "text_features": text_features,

            "fusion_features": fused_features

        }