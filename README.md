# TensorFlow Linear Regression Project

A beginner-friendly machine learning project that uses **TensorFlow 2** to train a simple linear regression model on synthetic data.

The model learns the relationship:

```text
y = 3x + 2
```

Noise is added to the data to make the task more realistic.

---

## Project Overview

This project demonstrates:

- Synthetic dataset generation using NumPy
- Building a simple neural network using TensorFlow/Keras
- Training a regression model
- Evaluating model performance
- Making predictions
- Visualizing actual data vs predicted line
- Saving the trained model

---

## Tech Stack

- Python
- TensorFlow 2
- NumPy
- Matplotlib
- scikit-learn

---

## Folder Structure

```text
tensorflow-linear-regression-project/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── outputs/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/tensorflow-linear-regression-project.git
cd tensorflow-linear-regression-project
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python src/train.py
```

After training, the project will:

- Train a TensorFlow model
- Print final loss and learned parameters
- Save the trained model in the `models/` folder
- Save the prediction plot in the `outputs/` folder

---

## Run Prediction

```bash
python src/predict.py
```

This loads the saved model and predicts output values for sample inputs.

---

## Expected Result

Since the original relationship is:

```text
y = 3x + 2
```

The model should learn values close to:

```text
weight ≈ 3
bias ≈ 2
```

Because random noise is added, the values may not be exactly 3 and 2.

---

## Sample Output

```text
Training completed successfully.
Final Loss: 0.23
Learned Weight: 2.98
Learned Bias: 2.07
Model saved at: models/linear_regression_model.keras
Plot saved at: outputs/regression_plot.png
```

---

## What I Learned

Through this project, I learned how to:

- Generate synthetic training data
- Build a simple TensorFlow regression model
- Train and evaluate a machine learning model
- Save and reload trained models
- Visualize predictions using Matplotlib

---

## Future Improvements

- Add command-line arguments for epochs and learning rate
- Add TensorBoard logging
- Use real-world regression datasets
- Deploy the model using Flask or FastAPI
- Add unit tests for data generation and prediction

---

## Author

**Nihal**  
Computer Science Graduate | Software Engineer | Python & Machine Learning Enthusiast
