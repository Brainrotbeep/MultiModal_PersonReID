import torch
import numpy as np

from models.resnet50model import ReIDModel

from engine.evaluate import (
    get_test_loaders,
    extract_features,
    compute_distance,
    evaluate
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model load
model = ReIDModel(num_classes=751).to(device)

# OPTIONAL (recommended if you saved model)
#model.load_state_dict(torch.load("checkpoint_epoch_20.pth"))

# loaders
query_loader, gallery_loader = get_test_loaders()

# extract features
qf, q_ids, q_cams = extract_features(query_loader, model, device)
gf, g_ids, g_cams = extract_features(gallery_loader, model, device)

# compute distance
distmat = compute_distance(qf, gf)

# 🔥 IMPORTANT FIX (Tensor → NumPy safe conversion)
distmat = distmat.cpu()

# evaluate
CMC, mAP = evaluate(distmat, q_ids, g_ids, q_cams, g_cams)

# results
print("Rank-1:", CMC[0])
print("Rank-5:", CMC[4])
print("Rank-10:", CMC[9])
print("mAP:", mAP)