import os

import torch
import torch.nn as nn
import csv
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import config
print("Reached train.py")
if config.IS_OCCLUDED:
    from dataset_occluded import PersonReIDDataset
else:
    from dataset import PersonReIDDataset

from model import MultimodalReID
from evaluate import evaluate


def train():
    print("Entered train()")
    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    dataset = PersonReIDDataset(
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
    print("\n=================================")
    print("Training Configuration")
    print("=================================")

    print(f"Dataset          : {config.DATASET}")
    print(f"Train Directory  : {config.TRAIN_DIR}")
    print(f"Caption File     : {config.TRAIN_CAPTIONS}")
    print(f"Checkpoint Dir   : {config.CHECKPOINT_DIR}")
    print(f"Classes          : {config.NUM_CLASSES}")
    print(f"Device           : {config.DEVICE}")
    print(f"Training Samples : {len(dataset)}")
    print(f"Batches          : {len(train_loader)}")
    print(f"Batch Size       : {config.BATCH_SIZE}")
    print(f"Epochs           : {config.EPOCHS}")
    print(f"Learning Rate    : {config.LEARNING_RATE}")

    print("=================================\n")
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_map = 0.0
    loss_history = []

    csv_file = os.path.join(
        config.CHECKPOINT_DIR,
        "training_loss.csv"
    )

    for epoch in range(config.EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.EPOCHS}"
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
                f"{100 * correct / total:.2f}%"

            })

        epoch_loss = (
                running_loss /
                len(train_loader)
        )

        epoch_acc = (
                correct /
                total
        )

        loss_history.append([
            epoch + 1,
            epoch_loss,
            epoch_acc
        ])

        print(f"\nEpoch [{epoch + 1}/{config.EPOCHS}]")
        print(f"Loss     : {epoch_loss:.4f}")
        print(f"Accuracy : {epoch_acc:.4f}")

        if (

            (epoch + 1) % 10 == 0

            or

            (epoch + 1) == config.EPOCHS

        ):

            print("\nRunning Evaluation...\n")

            rank1, rank5, rank10, mAP = evaluate(
                model
            )

            print(f"Rank-1  : {rank1:.4f}")
            print(f"Rank-5  : {rank5:.4f}")
            print(f"Rank-10 : {rank10:.4f}")
            print(f"mAP     : {mAP:.4f}")

            if mAP > best_map:

                best_map = mAP

                save_path = os.path.join(

                    config.CHECKPOINT_DIR,

                    config.MODEL_NAME.replace(
                        ".pth",
                        f"_best_epoch{epoch + 1}.pth"
                    )

                )

                torch.save(
                    model.state_dict(),
                    save_path
                )

                print(
                    f"\nBest model saved : {save_path}"
                )

            else:

                print(
                    "\nNo improvement."
                )

    final_path = os.path.join(
        config.CHECKPOINT_DIR,
        config.MODEL_NAME
    )

    torch.save(
        model.state_dict(),
        final_path
    )
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch",
            "Loss",
            "Accuracy"
        ])
        writer.writerows(loss_history)

    print(f"Loss history saved to {csv_file}")
    print("\n=================================")
    print("Training Finished")
    print("=================================\n")

    print(
        f"Best mAP    : {best_map:.4f}"
    )

    print(
        f"Final Model : {final_path}"
    )


if __name__ == "__main__":

    train()