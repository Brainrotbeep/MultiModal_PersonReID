import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

import config

print("Reached train.py")

if config.IS_OCCLUDED:
    from dataset_occluded import PersonReIDDataset
else:
    from dataset import PersonReIDDataset

from model import MultimodalReID
from evaluate import evaluate

# Bag of Tricks
from sampler import RandomIdentitySampler
from triplet_loss import BatchHardTripletLoss


def train():

    print("Entered train()")

    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = PersonReIDDataset(

        image_dir=config.TRAIN_DIR,

        caption_file=config.TRAIN_CAPTIONS,

        train=True

    )

    # --------------------------------------------------
    # PK Sampler
    # --------------------------------------------------

    sampler = RandomIdentitySampler(

        dataset,

        num_pids=config.P,

        num_instances=config.K

    )

    train_loader = DataLoader(

        dataset,

        batch_size=config.BATCH_SIZE,

        sampler=sampler,

        num_workers=0,

        drop_last=True

    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = MultimodalReID().to(
        config.DEVICE
    )
    # ---------------------------------------
    # Resume Training
    # ---------------------------------------


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
    print(f"P x K            : {config.P} x {config.K}")
    print(f"Epochs           : {config.EPOCHS}")
    print(f"Learning Rate    : {config.LEARNING_RATE}")

    print("=================================\n")

    # --------------------------------------------------
    # Loss Functions
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss(

        label_smoothing=config.LABEL_SMOOTHING

    )

    triplet_criterion = BatchHardTripletLoss(

        margin=config.TRIPLET_MARGIN

    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = AdamW(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY

    )
    #if os.path.exists(resume_path):
    #  optimizer.load_state_dict(
          #  checkpoint["optimizer_state_dict"]
        #)

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    scheduler = CosineAnnealingLR(

        optimizer,

        T_max=config.EPOCHS,

        eta_min=config.MIN_LR

    )

    best_map = 0.0
    # ==================================================
    # Training Loop
    # ==================================================

    for epoch in range( config.EPOCHS):

        model.train()

        running_loss = 0.0
        running_id_loss = 0.0
        running_triplet_loss = 0.0

        correct = 0
        total = 0

        progress = tqdm(

            train_loader,

            desc=f"Epoch {epoch + 1}/{config.EPOCHS}"

        )

        for batch in progress:

            images = batch["image"].to(config.DEVICE)

            input_ids = batch["input_ids"].to(config.DEVICE)

            attention_mask = batch["attention_mask"].to(config.DEVICE)

            labels = batch["label"].to(config.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(

                images,

                input_ids,

                attention_mask

            )

            # ------------------------------------------
            # Identity Loss
            # ------------------------------------------

            logits = outputs["logits"]

            id_loss = criterion(

                logits,

                labels

            )

            # ------------------------------------------
            # Triplet Loss
            # ------------------------------------------

            triplet_features = outputs["triplet_features"]

            triplet_loss = triplet_criterion(

                triplet_features,

                labels

            )

            # ------------------------------------------
            # Total Loss
            # ------------------------------------------

            loss = (

                config.ID_LOSS_WEIGHT * id_loss

                +

                config.TRIPLET_LOSS_WEIGHT * triplet_loss

            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                config.GRAD_CLIP

            )

            optimizer.step()

            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            running_loss += loss.item()

            running_id_loss += id_loss.item()

            running_triplet_loss += triplet_loss.item()

            preds = torch.argmax(

                logits,

                dim=1

            )

            correct += (

                preds == labels

            ).sum().item()

            total += labels.size(0)

            progress.set_postfix({

                "ID":

                f"{id_loss.item():.3f}",

                "Tri":

                f"{triplet_loss.item():.3f}",

                "Loss":

                f"{loss.item():.3f}",

                "Acc":

                f"{100*correct/total:.2f}%"

            })

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)

        epoch_id_loss = running_id_loss / len(train_loader)

        epoch_triplet_loss = running_triplet_loss / len(train_loader)

        epoch_acc = correct / total

        print("\n=================================")

        print(f"Epoch {epoch+1}/{config.EPOCHS}")

        print("=================================")

        print(f"Identity Loss : {epoch_id_loss:.4f}")

        print(f"Triplet Loss  : {epoch_triplet_loss:.4f}")

        print(f"Total Loss    : {epoch_loss:.4f}")

        print(f"Accuracy      : {epoch_acc:.4f}")

        print(

            f"Learning Rate : {optimizer.param_groups[0]['lr']:.8f}"

        )
        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        if (

            (epoch + 1) % config.EVAL_INTERVAL == 0

            or

            (epoch + 1) == config.EPOCHS

        ):

            rank1, rank5, rank10, mAP = evaluate(model)

            print("\n========== Evaluation ==========")

            print(f"Rank-1  : {rank1:.4f}")
            print(f"Rank-5  : {rank5:.4f}")
            print(f"Rank-10 : {rank10:.4f}")
            print(f"mAP     : {mAP:.4f}")

            print("================================\n")

            # ----------------------------------------------
            # Save Best Model (mAP)
            # ----------------------------------------------

            if mAP > best_map:

                best_map = mAP

                best_path = os.path.join(

                    config.CHECKPOINT_DIR,

                    "best_model.pth"

                )

                torch.save(

                    {

                        "epoch": epoch + 1,

                        "model_state_dict": model.state_dict(),

                        "optimizer_state_dict": optimizer.state_dict(),

                        "rank1": rank1,

                        "mAP": mAP,

                    },

                    best_path

                )

                print(f"Best model saved to {best_path}")

        # --------------------------------------------------
        # Save Checkpoint
        # --------------------------------------------------

        if (

            (epoch + 1) % config.SAVE_INTERVAL == 0

            or

            (epoch + 1) == config.EPOCHS

        ):

            checkpoint_path = os.path.join(

                config.CHECKPOINT_DIR,

                f"epoch_{epoch+1}.pth"

            )

            torch.save(

                {

                    "epoch": epoch + 1,

                    "model_state_dict": model.state_dict(),

                    "optimizer_state_dict": optimizer.state_dict(),

                },

                checkpoint_path

            )

            print(f"Checkpoint saved to {checkpoint_path}")

    # ==================================================
    # Save Final Model
    # ==================================================

    final_path = os.path.join(

        config.CHECKPOINT_DIR,

        "final_model.pth"

    )

    torch.save(

        model.state_dict(),

        final_path

    )

    print("\n=================================")

    print("Training Finished Successfully!")

    print(f"Final model saved to {final_path}")

    print("=================================\n")


if __name__ == "__main__":

    train()