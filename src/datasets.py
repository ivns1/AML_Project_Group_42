#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor

from config import MODEL_NAME

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)


class BirdTrainDataset(Dataset):
    def __init__(self, df, class_attr_matrix):
        self.df = df.reset_index(drop=True)
        self.class_attr_matrix = class_attr_matrix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        enc = processor(images=image, return_tensors="pt")

        item = {k: v.squeeze(0) for k, v in enc.items()}
        cls = int(row["label"]) - 1

        item["labels"] = torch.tensor(cls, dtype=torch.long)
        item["attributes"] = self.class_attr_matrix[cls]
        return item


class BirdTestDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        enc = processor(images=image, return_tensors="pt")

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["id"] = int(row["id"])
        return item

