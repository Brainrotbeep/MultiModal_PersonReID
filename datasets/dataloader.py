from torch.utils.data import DataLoader
from datasets.dataset import Market1501, transform


def get_dataloader(train_path, batch_size=32):
    dataset = Market1501(train_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0   # IMPORTANT for Windows
    )

    return loader