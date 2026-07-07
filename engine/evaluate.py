import torch
from torch.utils.data import DataLoader
from datasets.dataset import Market1501, transform

def get_test_loaders():
    query_path = "data/Market-1501/query"
    gallery_path = "data/Market-1501/bounding_box_test"

    query_set = Market1501(query_path, transform=transform)
    gallery_set = Market1501(gallery_path, transform=transform)

    query_loader = DataLoader(query_set, batch_size=32, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_set, batch_size=32, shuffle=False, num_workers=0)

    return query_loader, gallery_loader
def extract_features(loader, model, device):
    model.eval()

    features = []
    labels = []
    cams = []

    with torch.no_grad():
        for imgs, lbls, cam in loader:
            imgs = imgs.to(device)

            feat, _ = model(imgs)

            feat = torch.nn.functional.normalize(feat, p=2, dim=1)

            features.append(feat.cpu())
            labels.extend(lbls)
            cams.extend(cam)

    return torch.cat(features), labels, cams
def compute_distance(qf, gf):
    return torch.cdist(qf, gf)
import numpy as np

def evaluate(distmat, q_pids, g_pids, q_cams, g_cams):
    import numpy as np

    q_pids = np.array(q_pids)
    g_pids = np.array(g_pids)
    q_cams = np.array(q_cams)
    g_cams = np.array(g_cams)

    num_q = distmat.shape[0]
    indices = np.argsort(distmat.numpy(), axis=1)

    CMC = np.zeros(len(g_pids))
    APs = []

    for i in range(num_q):
        q_pid = q_pids[i]
        q_cam = q_cams[i]

        order = indices[i]

        remove = (g_pids == q_pid) & (g_cams == q_cam)
        keep = np.invert(remove)

        matches = (g_pids[order] == q_pid).astype(int)
        matches = matches[keep[order]]

        if not np.any(matches):
            continue

        first_idx = np.where(matches == 1)[0][0]
        CMC[first_idx:] += 1

        cum = matches.cumsum()
        precision = cum / (np.arange(len(matches)) + 1)
        AP = (precision * matches).sum() / matches.sum()
        APs.append(AP)

    CMC = CMC / num_q
    mAP = np.mean(APs)

    return CMC, mAP