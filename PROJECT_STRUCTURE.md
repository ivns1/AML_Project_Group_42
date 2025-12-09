# AML Project - Feathers in Focus

## Project Overview

This project is a group assignment for the **UvA (Universiteit van Amsterdam) Applied Machine Learning** course. The task is bird species classification in a Kaggle competition format.

### Project Information
| Property | Value |
|----------|-------|
| **Course** | UvA Applied Machine Learning |
| **Project Name** | Feathers in Focus |
| **Task** | Bird Species Classification (Image Classification) |
| **Number of Classes** | 200 different bird species |
| **Kaggle Link** | https://www.kaggle.com/t/0e9856f5cb5f40af8739be017cc75b9b |

---

## Dataset Summary

### Statistics
| Metric | Value |
|--------|-------|
| **Training Images** | 3,926 |
| **Test Images** | 4,000 |
| **Total Classes** | 200 bird species |
| **Number of Attributes** | 312 features |
| **Training Data Size** | ~369 MB |
| **Test Data Size** | ~379 MB |

### Average Images per Class
- Training: ~19.6 images/class
- Test: 20 images/class

---

## Directory Structure

```
AML_Project/
├── aml-2025-feathers-in-focus/     # Main data folder
│   ├── train_images/               # Training images
│   │   └── train_images/
│   │       └── *.jpg               # 3,926 JPG files
│   ├── test_images/                # Test images
│   │   └── test_images/
│   │       └── *.jpg               # 4,000 JPG files
│   ├── train_images.csv            # Training labels
│   ├── test_images_sample.csv      # Submission example
│   ├── test_images_path.csv        # Test file paths
│   ├── class_names.npy             # Class names (numpy)
│   ├── attributes.npy              # Attribute matrix (numpy)
│   └── attributes.txt              # Attribute names (text)
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── dataset.py                  # Dataset and dataloader
│   ├── losses.py                   # Loss functions (multi-task)
│   ├── metrics.py                  # Evaluation metrics
│   ├── model.py                    # Model architectures
│   ├── trainer.py                  # Training loop
│   ├── transforms.py               # Data augmentation
│   └── utils.py                    # Utility functions
│
├── train.py                        # Main training script
├── inference.py                    # Inference and submission generation
├── class_names.csv                 # Processed class names
├── attribute_names.csv             # Processed attribute names
├── attributes.csv                  # Processed attribute matrix
├── Preprocessing.py                # Data preprocessing script
├── Group_project_description.pdf   # Project description
└── README.md                       # Basic information
```

---

## File Formats and Contents

### 1. train_images.csv
Contains training data labels.

| Column | Description |
|--------|-------------|
| `image_path` | Image file path (e.g., `/train_images/1.jpg`) |
| `label` | Class label (1-200) |

**Example:**
```csv
image_path,label
/train_images/1.jpg,1
/train_images/2.jpg,1
/train_images/3.jpg,1
```

### 2. test_images_sample.csv
Kaggle submission format. **Output file must follow this format!**

| Column | Description |
|--------|-------------|
| `id` | Test sample ID (1-4000) |
| `label` | Predicted class (1-200) |

**Example:**
```csv
id,label
1,1
2,1
3,1
```

### 3. class_names.csv
Names of 200 bird species.

| Column | Description |
|--------|-------------|
| `Label` | Class number (1-200) |
| `Class_name` | Bird species name |

**Example Classes:**
| Label | Class_name |
|-------|------------|
| 1 | Black_footed_Albatross |
| 2 | Laysan_Albatross |
| 17 | Cardinal |
| 200 | Common_Yellowthroat |

### 4. attribute_names.csv
Definitions of 312 different attributes.

| Column | Description |
|--------|-------------|
| `id` | Attribute ID (1-312) |
| `attribute_group` | Attribute group |
| `attribute` | Attribute value |

**Attribute Groups:**
- `has_bill_shape`: Bill shape (curved, dagger, hooked, needle, cone, etc.)
- `has_wing_color`: Wing color (blue, brown, grey, yellow, etc.)
- `has_upperparts_color`: Upperparts color
- `has_underparts_color`: Underparts color
- `has_breast_pattern`: Breast pattern
- `has_back_color`: Back color
- `has_tail_shape`: Tail shape
- `has_upper_tail_color`: Upper tail color
- `has_head_pattern`: Head pattern
- `has_breast_color`: Breast color
- `has_throat_color`: Throat color
- `has_eye_color`: Eye color
- `has_bill_length`: Bill length
- `has_forehead_color`: Forehead color
- `has_under_tail_color`: Under tail color
- `has_nape_color`: Nape color
- `has_belly_color`: Belly color
- `has_wing_shape`: Wing shape
- `has_size`: Size
- `has_shape`: Overall shape
- `has_back_pattern`: Back pattern
- `has_tail_pattern`: Tail pattern
- `has_belly_pattern`: Belly pattern
- `has_primary_color`: Primary color
- `has_leg_color`: Leg color
- `has_bill_color`: Bill color
- `has_crown_color`: Crown color
- `has_wing_pattern`: Wing pattern

### 5. attributes.csv
Probability values (0.0 - 1.0) for 312 attributes for each class.

| Column | Description |
|--------|-------------|
| `Class_id` | Class number (1-200) |
| `1-312` | Probability score for each attribute |

**Dimensions:** 200 rows x 313 columns (Class_id + 312 attributes)

---

## Preprocessing.py Script

This script converts raw numpy format data to CSV format:

### Functions:
1. **class_names.npy -> class_names.csv**
   - Converts class names to readable CSV format

2. **attributes.txt -> attribute_names.csv**
   - Separates attribute definitions into ID, group, and name

3. **attributes.npy -> attributes.csv**
   - Converts attribute matrix for each class to CSV

### Usage:
```bash
python Preprocessing.py
```

---

## Project Requirements

### Tasks:
1. **Baseline Model**: Create a baseline using a pretrained model from HuggingFace
   - Submit to Kaggle with the name "baseline"

2. **Custom Model**: Develop your own ML model and try to exceed the baseline

3. **Analysis**:
   - Computational complexity comparison
   - Error analysis (where the model makes incorrect predictions)

### Deliverables:
- **Poster**: Main findings, model architecture, results
- **Kaggle Submission**: In test_images_sample.csv format

### Poster Content:
- Main research question
- Model architecture figure
- Baseline comparison
- Computational complexity analysis
- Error analysis (1-2 examples)

---

## Suggested Approaches

### 1. Baseline (HuggingFace Pretrained)
- Pretrained models like ResNet, EfficientNet, ViT
- Fine-tuning for bird classification adaptation

### 2. Custom Model Ideas
- CNN-based architecture (with fewer parameters)
- Using attribute information (multi-task learning)
- Small model with transfer learning
- Data augmentation techniques

### 3. Attribute Usage
The attributes.csv file provides additional information:
- Visual features are defined for each class
- This information can be used as auxiliary loss
- Can be useful for zero-shot or few-shot learning

---

## Quick Start

### 1. Data Loading (PyTorch)
```python
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.root_dir + self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label'] - 1  # 0-indexed

        if self.transform:
            image = self.transform(image)

        return image, label
```

### 2. Loading Class Names
```python
import pandas as pd

class_names = pd.read_csv('class_names.csv')
label_to_name = dict(zip(class_names['Label'], class_names['Class_name']))

# Example: label_to_name[1] = 'Black_footed_Albatross'
```

### 3. Loading Attribute Information
```python
import pandas as pd

attributes = pd.read_csv('attributes.csv')
attr_names = pd.read_csv('attribute_names.csv')

# Attribute vector for class 1
class_1_attrs = attributes[attributes['Class_id'] == 1].iloc[0, 1:].values
```

---

## Important Notes

1. **Submission Format**: Must be in the same format as test_images_sample.csv
2. **Label Range**: 1-200 (not 0-199!)
3. **Baseline Naming**: Name it "baseline" when submitting to Kaggle
4. **Computational Cost**: Computational cost is as important as model performance

---

## File Paths Reference

| File | Path |
|------|------|
| Training CSV | `aml-2025-feathers-in-focus/train_images.csv` |
| Test Sample | `aml-2025-feathers-in-focus/test_images_sample.csv` |
| Training Images | `aml-2025-feathers-in-focus/train_images/train_images/*.jpg` |
| Test Images | `aml-2025-feathers-in-focus/test_images/test_images/*.jpg` |
| Class Names | `class_names.csv` |
| Attribute Names | `attribute_names.csv` |
| Attribute Matrix | `attributes.csv` |

---

## Contact and Resources

- **Kaggle Competition**: https://www.kaggle.com/t/0e9856f5cb5f40af8739be017cc75b9b
- **HuggingFace Models**: https://huggingface.co/models?pipeline_tag=image-classification
