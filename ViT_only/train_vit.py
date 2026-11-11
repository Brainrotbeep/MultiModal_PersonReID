import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import config_vit

if (config_vit
        .IS_OCCLUDED):
    from dataset_occluded_vit import PersonReIDDataset
else:
    from dataset_vit import PersonReIDDataset

from model_vit import ViTBaseline
from evaluate_vit import evaluate

def train():

    os.makedirs(
        config_vit.CHECKPOINT_DIR,
        exist_ok=True
    )

    dataset = PersonReIDDataset(

        image_dir=config_vit.TRAIN_DIR,

        train=True

    )

    train_loader = DataLoader(

        dataset,

        batch_size=config_vit.BATCH_SIZE,

        shuffle=True,

        num_workers=0

    )

    model = ViTBaseline().to(

        config_vit.DEVICE

    )

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(

        model.parameters(),

        lr=config_vit.LEARNING_RATE,

        weight_decay=config_vit.WEIGHT_DECAY

    )

    best_map = 0.0

    print("\n=================================")
    print("ViT Baseline Training")
    print("=================================")

    print(f"Dataset          : {config_vit.DATASET}")
    print(f"Training Samples : {len(dataset)}")
    print(f"Batches          : {len(train_loader)}")
    print(f"Classes          : {config_vit.NUM_CLASSES}")
    print(f"Epochs           : {config_vit.EPOCHS}")
    print(f"Device           : {config_vit.DEVICE}")

    print("=================================\n")

    for epoch in range(config_vit.EPOCHS):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(

            train_loader,

            desc=f"Epoch {epoch+1}/{config_vit.EPOCHS}"

        )

        for batch in progress:

            images = batch["image"].to(

                config_vit.DEVICE

            )

            labels = batch["label"].to(

                config_vit.DEVICE

            )

            optimizer.zero_grad(

                set_to_none=True

            )

            outputs = model(

                images

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

                "Loss": f"{loss.item():.4f}",

                "Acc": f"{100*correct/total:.2f}%"

            })

        epoch_loss = running_loss / len(train_loader)

        epoch_acc = correct / total

        print(f"\nEpoch {epoch+1}")

        print(f"Loss     : {epoch_loss:.4f}")

        print(f"Accuracy : {epoch_acc:.4f}")

        if (

            (epoch+1) % 10 == 0

            or

            (epoch+1) == config_vit.EPOCHS

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
                    config_vit.CHECKPOINT_DIR,
                    config_vit.BEST_MODEL_NAME
                )

                torch.save(

                    model.state_dict(),

                    save_path

                )

                print(

                    f"\nBest Model Saved : {save_path}"

                )

    final_path = os.path.join(
        config_vit.CHECKPOINT_DIR,
        config_vit.MODEL_NAME
    )

    torch.save(

        model.state_dict(),

        final_path

    )

    print("\n=================================")
    print("Training Finished")
    print("=================================")

    print(f"Best mAP    : {best_map:.4f}")
    print(f"Final Model : {final_path}")


if __name__ == "__main__":

    train()