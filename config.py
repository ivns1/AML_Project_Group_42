#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA
TRAIN_CSV = PROJECT_ROOT / "train_images.csv"
TEST_PATH_CSV = PROJECT_ROOT / "test_images_path.csv"
SAMPLE_SUB_CSV = PROJECT_ROOT / "test_images_sample.csv"

TRAIN_IMG_DIR = PROJECT_ROOT / "train_images" / "train_images"
TEST_IMG_DIR = PROJECT_ROOT / "test_images" / "test_images"

ATTRIBUTES_NPY = PROJECT_ROOT / "attributes.npy"
CLASS_NAMES_NPY = PROJECT_ROOT / "class_names.npy"

# OUTPUTS
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
OUTPUTS_ROOT.mkdir(exist_ok=True)

# MODEL
MODEL_NAME = "facebook/convnext-tiny-224"
NUM_CLASSES = 200
LATENT_DIM = 128

# TRAINING
TOTAL_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

LABEL_SMOOTHING = 0.1
ATTR_W_INIT = 0.5
ATTR_W_FINAL = 0.1
CONS_WEIGHT = 0.1

RANDOM_SEED = 42

