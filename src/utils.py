"""
Utility functions for the TensorFlow Linear Regression project.
"""

from pathlib import Path
import numpy as np


def create_directories() -> None:
    """
    Create required project directories if they do not exist.
    """
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)


def generate_synthetic_data(
    start: float = 0,
    stop: float = 10,
    samples: int = 100,
    noise_level: float = 0.5,
    seed: int = 42,
):
    """
    Generate synthetic linear regression data based on y = 3x + 2.

    Args:
        start: Starting value of X.
        stop: Ending value of X.
        samples: Number of data points.
        noise_level: Amount of random noise added to y.
        seed: Random seed for reproducibility.

    Returns:
        X: Input values reshaped for TensorFlow.
        y: Target values.
    """
    np.random.seed(seed)

    X = np.linspace(start, stop, samples)
    noise = np.random.randn(*X.shape) * noise_level
    y = 3 * X + 2 + noise

    return X.reshape(-1, 1), y
