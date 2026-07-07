import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import config

from dataset import Market1501Dataset
from model import MultimodalReID


def train():

    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    dataset = Market1501Dataset(
        image_dir=config.TRAIN_DIR,
        caption_file=config.TRAIN_CAPTIONS,
        train=True
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = MultimodalReID().to(
        config.DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_acc = 0.0

    print(
        f"\nDevice : {config.DEVICE}"
    )

    print(
        f"Training Samples : {len(dataset)}"
    )

    print(
        f"Batches : {len(train_loader)}\n"
    )

    for epoch in range(config.EPOCHS):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.EPOCHS}"
        )

        for batch in progress:

            images = batch["image"].to(
                config.DEVICE
            )

            input_ids = batch["input_ids"].to(
                config.DEVICE
            )

            attention_mask = batch["attention_mask"].to(
                config.DEVICE
            )

            labels = batch["label"].to(
                config.DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(

                images,

                input_ids,

                attention_mask

            )

            logits = outputs["logits"]

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix({

                "Loss":
                f"{loss.item():.4f}",

                "Acc":
                f"{100*correct/total:.2f}%"

            })

        epoch_loss = (
            running_loss /
            len(train_loader)
        )

        epoch_acc = (
            correct /
            total
        )

        print()

        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}]"
        )

        print(
            f"Loss     : {epoch_loss:.4f}"
        )

        print(
            f"Accuracy : {epoch_acc:.4f}"
        )

        if epoch_acc > best_acc:

            best_acc = epoch_acc

            torch.save(

                model.state_dict(),

                os.path.join(

                    config.CHECKPOINT_DIR,

                    "best_model.pth"

                )

            )

            print(
                "Best model saved.\n"
            )

    torch.save(

        model.state_dict(),

        os.path.join(

        config.CHECKPOINT_DIR,

        config.MODEL_NAME

        )

    )

    print(
        "\nTraining Finished."
    )

    print(
        f"Best Accuracy : {best_acc:.4f}"
    )

    print(
        "Final model saved."
    )


if __name__ == "__main__":

    train()