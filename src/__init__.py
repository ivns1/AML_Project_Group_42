# Bird Classification - Custom CNN Model
# UvA Applied Machine Learning Project

from .config import Config
from .dataset import BirdDataset, create_dataloaders
from .model import BirdClassifier
from .trainer import Trainer
from .losses import CombinedLoss
from .metrics import calculate_metrics

__version__ = "1.0.0"
__author__ = "AML Team"
