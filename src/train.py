"""
Train a simple TensorFlow 2 linear regression model.

The model learns the relationship:

    y = 3x + 2

using synthetic data with noise.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from utils import create_directories, generate_synthetic_data


def build_model() -> tf.keras.Model:
    """
    Build a simple sequential model with one dense layer.

    Returns:
        Compiled TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="mse"
    )

    return model


def plot_results(X, y, predictions) -> None:
    """
    Plot actual data and model predictions.

    Args:
        X: Input values.
        y: Actual target values.
        predictions: Model predicted values.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Training Data")
    plt.plot(X, predictions, label="Model Prediction")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression with TensorFlow 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/regression_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    """
    Main function to train and save the model.
    """
    create_directories()

    X, y = generate_synthetic_data()

    model = build_model()

    history = model.fit(
        X,
        y,
        epochs=200,
        verbose=0
    )

    predictions = model.predict(X, verbose=0).flatten()

    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    weight = model.layers[0].get_weights()[0][0][0]
    bias = model.layers[0].get_weights()[1][0]

    model.save("models/linear_regression_model.keras")

    print("Training completed successfully.")
    print(f"Final Loss: {history.history['loss'][-1]:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Learned Weight: {weight:.4f}")
    print(f"Learned Bias: {bias:.4f}")
    print("Model saved at: models/linear_regression_model.keras")
    print("Plot saved at: outputs/regression_plot.png")

    plot_results(X, y, predictions)


if __name__ == "__main__":
    main()
