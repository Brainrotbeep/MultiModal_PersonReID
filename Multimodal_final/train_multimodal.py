import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from Multimodal_dataset import (
    MultimodalDataset
)

from Multimodal_model import (
    MultimodalModel
)


device = torch.device(
    "cpu"
)

dataset = MultimodalDataset()

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

num_classes = len(
    set(dataset.labels)
)

model = MultimodalModel(
    num_classes
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

epochs = 20

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for features, labels in loader:

        features = features.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

        preds = outputs.argmax(
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    acc = (
        correct / total
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Acc: {acc:.4f}"
    )

torch.save(
    model.state_dict(),
    "multimodal_model.pth"
)

print(
    "\nSaved multimodal_model.pth"
)