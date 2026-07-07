import torch
import torch.nn as nn

from datasets.dataloader import get_dataloader
from models.resnet50model import ReIDModel

from engine.evaluate import (
    get_test_loaders,
    extract_features,
    compute_distance,
    evaluate
)

def train():
    print("🚀 Training started")

    train_path = "data/Market-1501/bounding_box_train"

    train_loader = get_dataloader(train_path)

    print("Dataset size:", len(train_loader.dataset))
    print("Batches:", len(train_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # number of identities
    num_classes = len(set(train_loader.dataset.labels))

    # model
    model = ReIDModel(num_classes=num_classes).to(device)

    # loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # test loaders
    query_loader, gallery_loader = get_test_loaders()

    # training loop
    for epoch in range(50):

        model.train()
        total_loss = 0

        for imgs, labels, _ in train_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            # forward
            _, outputs = model(imgs)

            # loss
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1}/50")
        print(f"Loss: {total_loss:.4f}")

        # EVERY 10 EPOCHS → SAVE + EVALUATE

        if (epoch + 1) % 2 == 0:

            print("\n🔥 Running Evaluation...")

            # save checkpoint
            checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth"

            torch.save(model.state_dict(), checkpoint_name)

            print(f"✅ Model saved: {checkpoint_name}")

            # feature extraction
            qf, q_ids, q_cams = extract_features(
                query_loader,
                model,
                device
            )

            gf, g_ids, g_cams = extract_features(
                gallery_loader,
                model,
                device
            )

            # distance matrix
            distmat = compute_distance(qf, gf)

            distmat = distmat.cpu()

            # evaluation
            CMC, mAP = evaluate(
                distmat,
                q_ids,
                g_ids,
                q_cams,
                g_cams
            )

            print("\n📊 Evaluation Results")
            print(f"Rank-1 : {CMC[0]:.4f}")
            print(f"Rank-5 : {CMC[4]:.4f}")
            print(f"Rank-10: {CMC[9]:.4f}")
            print(f"mAP    : {mAP:.4f}")

    # final model save
    torch.save(model.state_dict(), "models/final_model.pth")

    print("\n✅ Final model saved: final_model.pth")


if __name__ == "__main__":
    train()