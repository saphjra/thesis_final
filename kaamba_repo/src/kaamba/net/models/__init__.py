"""
Neural network models for KAAMBA
"""

from kaamba.net.models.kaamba import GazePredictor
from kaamba.net.models.kaamba import (
    GazePredictor as KaambaGazePredictor,
    build_gaze_predictor,
)
from kaamba.net.models.kaamba_categorical import (
    GazeCategoricalPredictor,
    build_categorical_gaze_predictor,
)

__all__ = [
    "GazePredictor",
    "KaambaGazePredictor",
    "build_gaze_predictor",
    "GazeCategoricalPredictor",
    "build_categorical_gaze_predictor",
]
