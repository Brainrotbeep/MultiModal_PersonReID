import torch.nn as nn


class MultimodalModel(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                2432,
                1024
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                1024,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):

        return self.classifier(
            x
        )