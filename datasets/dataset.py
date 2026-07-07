import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class Market1501(Dataset):
    def __init__(self, root, transform=None):
        self.img_paths = []
        self.labels = []
        self.cams = []
        self.transform = transform

        pids = set()

        for img in os.listdir(root):
            if not img.endswith('.jpg'):
                continue

            pid = int(img.split('_')[0])
            if pid == -1:
                continue

            cam = int(img.split('_')[1][1])

            self.img_paths.append(os.path.join(root, img))
            self.labels.append(pid)
            self.cams.append(cam)

            pids.add(pid)

        # remap labels
        # remap ONLY for training set
        if "bounding_box_train" in root:
            self.pid2label = {
                pid: idx
                for idx, pid in enumerate(sorted(pids))
            }

            self.labels = [
                self.pid2label[pid]
                for pid in self.labels
            ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx], self.cams[idx]


# transforms
transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])