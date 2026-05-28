"""
KAAMBA: Gaze Prediction and Analysis Package

A comprehensive package for eye gaze prediction and analysis using modern
deep learning architectures (Mamba-based models).
"""

from kaamba.net.models.kaamba import GazePredictor
from kaamba.utils.loss_functions import gaussian_nll
from kaamba.utils.on_the_fly_dataset import create_on_the_fly_loader
from kaamba.utils.memory_monitor import MemoryMonitor, memory_tracker, get_summary


__version__ = "0.1.0"

__all__ = [
    "GazePredictor",
    "gaussian_nll",
    "create_on_the_fly_loader",
    "MemoryMonitor",
    "memory_tracker",
    "get_summary",
]
