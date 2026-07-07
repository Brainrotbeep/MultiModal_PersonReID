import torch.nn as nn
import torchvision.models as models

class ReIDModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.base = models.resnet50(pretrained=True)
        self.base.fc = nn.Identity()   # remove last layer

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.base(x)
        logits = self.classifier(feat)
        return feat, logits