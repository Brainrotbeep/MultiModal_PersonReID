import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss

    Reference:
    In Defense of the Triplet Loss for Person Re-Identification
    Bag of Tricks and A Strong Baseline for Deep Person Re-identification
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features, labels):

        # L2 Normalize features
        features = F.normalize(features, p=2, dim=1)

        batch_size = features.size(0)

        # Pairwise Euclidean distance
        dist = torch.cdist(features, features, p=2)

        labels = labels.view(-1)

        # Positive / Negative masks
        is_pos = labels.expand(batch_size, batch_size).eq(
            labels.expand(batch_size, batch_size).t()
        )

        is_neg = ~is_pos

        hardest_positive = []
        hardest_negative = []

        for i in range(batch_size):

            pos_dist = dist[i][is_pos[i]]
            neg_dist = dist[i][is_neg[i]]

            # Remove self-distance (=0)
            pos_dist = pos_dist[pos_dist > 1e-12]

            if len(pos_dist) == 0 or len(neg_dist) == 0:
                continue

            hardest_positive.append(pos_dist.max())
            hardest_negative.append(neg_dist.min())

        if len(hardest_positive) == 0:
            return torch.tensor(
                0.0,
                device=features.device,
                requires_grad=True
            )

        hardest_positive = torch.stack(hardest_positive)
        hardest_negative = torch.stack(hardest_negative)

        target = torch.ones_like(hardest_negative)

        loss = self.ranking_loss(
            hardest_negative,
            hardest_positive,
            target
        )

        return loss