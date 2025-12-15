# ConvNeXt Multi-Task Model

This branch contains an implementation of a ConvNeXt-Tiny multi-task learning model
for bird species classification with attribute supervision.

This branch is intended to be compared against a custom CNN baseline
implemented in a separate branch.

---

## Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── datasets.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

## Setup

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

To train the ConvNeXt multi-task model, run:

```bash
python src/train.py
```

During training:
- The dataset is split into training and validation sets
- Training loss and validation accuracy are printed to the console
- The trained model is saved to:

```text
outputs/model.pt
```

---

## Inference and Submission Generation

After training, generate predictions for the test set:

```bash
python src/inference.py
```

This will:
- Load the trained model from `outputs/model.pt`
- Run inference on the test images
- Generate a submission file at:

```text
outputs/submission_convnext_multitask.csv
```

The submission format matches `test_images_sample.csv`.

---

## Configuration

All dataset paths and hyperparameters are defined in:

```text
src/config.py
```

This file controls:
- Dataset locations
- Model configuration
- Training hyperparameters
- Output directories
