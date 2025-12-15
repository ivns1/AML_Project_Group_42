import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import *
from datasets import BirdTrainDataset
from model import ConvNeXtMultiTask
from utils import set_seed


def main():
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # LOAD DATA
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["image_path"] = train_df["image_path"].str.lstrip("/")
    train_df["image_path"] = train_df["image_path"].apply(
        lambda p: TRAIN_IMG_DIR / p.split("/")[-1]
    )

    attributes = np.load(ATTRIBUTES_NPY)
    class_attr_matrix = torch.tensor(attributes, dtype=torch.float32)

    train_df_, val_df_ = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=RANDOM_SEED,
    )

    train_ds = BirdTrainDataset(train_df_, class_attr_matrix)
    val_ds = BirdTrainDataset(val_df_, class_attr_matrix)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = ConvNeXtMultiTask(class_attr_matrix).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # TRAINING
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            attributes_batch = batch["attributes"].to(device)

            loss, _ = model(
                pixel_values,
                labels=labels,
                attributes=attributes_batch,
                epoch=epoch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * pixel_values.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # VALIDATION
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                logits = model(pixel_values)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(
            f"Epoch {epoch+1}/{TOTAL_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), OUTPUTS_ROOT / "model.pt")
    print("Model saved.")


if __name__ == "__main__":
    main()
