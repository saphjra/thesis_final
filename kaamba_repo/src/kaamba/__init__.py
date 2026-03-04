"""
KAAMBA: Gaze Prediction and Analysis Package

A comprehensive package for eye gaze prediction and analysis using modern
deep learning architectures (Mamba-based models).
"""

from kaamba.net.models.tamba import GazePredictor
from kaamba.utils.loss_functions import gaussian_nll
from kaamba.utils.on_the_fly_dataset import create_on_the_fly_loader
from kaamba.utils.memory_monitor import MemoryMonitor, memory_tracker, get_summary
from kaamba.utils import constants
from kaamba.scripts.train_on_the_fly import train_on_the_fly

__version__ = "0.1.0"

__all__ = [
    "GazePredictor",
    "gaussian_nll",
    "create_on_the_fly_loader",
    "MemoryMonitor",
    "memory_tracker",
    "train_on_the_fly",
    "get_summary"
]
