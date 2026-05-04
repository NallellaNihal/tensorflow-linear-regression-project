"""
Load the trained TensorFlow model and make predictions.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf


MODEL_PATH = Path("models/linear_regression_model.keras")


def main() -> None:
    """
    Load model and predict y values for sample X values.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Please run `python src/train.py` first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    sample_inputs = np.array([[0], [2], [5], [10], [15]], dtype=float)
    predictions = model.predict(sample_inputs, verbose=0).flatten()

    print("Predictions:")
    for x_value, prediction in zip(sample_inputs.flatten(), predictions):
        print(f"x = {x_value:.2f}  ->  predicted y = {prediction:.2f}")


if __name__ == "__main__":
    main()
