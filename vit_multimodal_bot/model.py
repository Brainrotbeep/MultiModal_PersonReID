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

            nn.ReLU(inplace=True),

            nn.Dropout(
                config.DROPOUT
            )

        )

        # --------------------------------------------------
        # BNNeck
        # --------------------------------------------------

        self.bnneck = nn.BatchNorm1d(
            config.FUSION_DIM
        )

        # No bias learning in BNNeck
        self.bnneck.bias.requires_grad_(False)

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Linear(

            config.FUSION_DIM,

            config.NUM_CLASSES,

            bias=False

        )

        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------

        nn.init.xavier_uniform_(
            self.fusion[0].weight
        )

        nn.init.zeros_(
            self.fusion[0].bias
        )

        nn.init.normal_(
            self.classifier.weight,
            std=0.001
        )

    def forward(

        self,

        images,

        input_ids,

        attention_mask

    ):

        # --------------------------------------------------
        # Image Features
        # --------------------------------------------------

        image_outputs = self.image_encoder(
            pixel_values=images
        )

        image_features = image_outputs.last_hidden_state[
            :, 0
        ]

        # --------------------------------------------------
        # Text Features
        # --------------------------------------------------

        text_outputs = self.text_encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        text_features = text_outputs.last_hidden_state[
            :, 0
        ]

        # --------------------------------------------------
        # Weighted Feature Fusion
        # --------------------------------------------------

        IMAGE_WEIGHT = 0.75
        TEXT_WEIGHT = 0.25

        fusion_input = torch.cat(

            [

                image_features * IMAGE_WEIGHT,

                text_features * TEXT_WEIGHT

            ],

            dim=1

        )

        # Raw embedding
        triplet_features = self.fusion(
            fusion_input
        )

        # BNNeck feature
        bn_features = self.bnneck(
            triplet_features
        )

        # Identity classifier
        logits = self.classifier(
            bn_features
        )

        return {

            "logits": logits,

            "triplet_features": triplet_features,

            "bn_features": bn_features,

            "image_features": image_features,

            "text_features": text_features

        }