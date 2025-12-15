import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from config import *
from datasets import BirdTestDataset
from model import ConvNeXtMultiTask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(TEST_PATH_CSV)
    test_df["image_path"] = test_df["image_path"].str.lstrip("/")
    test_df["image_path"] = test_df["image_path"].apply(
        lambda p: TEST_IMG_DIR / p.split("/")[-1]
    )

    attributes = np.load(ATTRIBUTES_NPY)
    class_attr_matrix = torch.tensor(attributes, dtype=torch.float32)

    test_ds = BirdTestDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvNeXtMultiTask(class_attr_matrix).to(device)
    model.load_state_dict(
        torch.load(OUTPUTS_ROOT / "model.pt", map_location=device)
    )
    model.eval()

    preds, ids = [], []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            logits = model(pixel_values)
            batch_preds = logits.argmax(dim=-1) + 1
            preds.extend(batch_preds.cpu().tolist())
            ids.extend(batch["id"])

    submission = pd.read_csv(SAMPLE_SUB_CSV)
    submission["label"] = preds
    submission.to_csv(
        OUTPUTS_ROOT / "submission_convnext_multitask.csv", index=False
    )

    print("Submission saved.")


if __name__ == "__main__":
    main()
