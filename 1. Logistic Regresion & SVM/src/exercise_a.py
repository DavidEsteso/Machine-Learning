import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def load_data(filepath):
    """Load data from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def visualize_data(df):
    """Visualize the data in a 2D plot."""
    X1 = df.iloc[:, 0]
    X2 = df.iloc[:, 1]
    y = df.iloc[:, 2]

    plt.figure(figsize=(10, 6))
    plt.scatter(X1[y == 1], X2[y == 1], marker='+', color='purple', label='Training +1')
    plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='orange', label='Training -1')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    plt.savefig('../output/a/exercise_i.png')
    plt.close()

def visualize_predictions_with_decision_boundary(df, model):
    """Visualize training data, predictions, and decision boundary of the model."""
    features = df.shape[1] - 1
    X1 = df.iloc[:, 0].values
    X2 = df.iloc[:, 1].values
    y = df.iloc[:, features].values
    
    # Make predictions using the model
    predictions = model.predict(df.iloc[:, :features].values)
    c = model.intercept_[0]
    m = model.coef_[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(X1[y == 1], X2[y == 1], marker='+', color='purple', label='Training +1')
    plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='orange', label='Training -1')
    plt.scatter(X1[predictions == 1], X2[predictions == 1], marker='x', color='pink', label='Predicted +1')
    plt.scatter(X1[predictions == -1], X2[predictions == -1], marker='x', color='yellow', label='Predicted -1')

    if features == 2:
        x_values = np.linspace(X1.min(), X1.max(), 100)
        # Solve for y values using the decision boundary equation: c + m1*x + m2*y = 0
        y_values = -(c + m[0] * x_values) / m[1]
        plt.plot(x_values, y_values, color='black', label='Decision Boundary')
    elif features == 4:
        xx1, xx2 = np.meshgrid(np.linspace(X1.min() + 0.25, X1.max() + 0.1, 100), np.linspace(X2.min() + 0.25, X2.max() + 0.25, 100))

        # Decision boundary equation: w0 + w1*x1 + w2*x2 + w3*x1^2 + w4*x2^2 = 0
        decision_boundary = (m[0] * xx1 + m[1] * xx2 + c + m[2] * (xx1 ** 2) + m[3] * (xx2 ** 2))
        plt.contour(xx1, xx2, decision_boundary, levels=[0], colors='black')
        # Auxiliar line in order to represent the decision boundary in the legend (contour plot does not have a legend)
        plt.plot([], [], 'k-', label='Decision Boundary', color='black')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    filename = '../output/a/exercise_iii.png' if features == 2 else '../output/c/exercise_ii.png'
    plt.savefig(filename)
    plt.close()

def train_logistic_regression(df):
    """Train a logistic regression model and save feature influence analysis."""
    features = df.shape[1] - 1
    X = df.iloc[:, :features].values
    y = df.iloc[:, features].values

    # Train a logistic regression model
    model = LogisticRegression().fit(X, y)

    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    filename = '../output/a/exercise_ii.txt' if features == 2 else '../output/c/exercise_i.txt'
    with open(filename, 'w') as f:
        f.write("Logistic Regression Model Parameters:\n")
        f.write(f"Coefficients: {coefficients}\n")
        f.write(f"Intercept: {intercept}\n\n")
        f.write("Additional Model Information:\n")
        f.write(f"Number of Iterations: {model.n_iter_}\n")
        f.write(f"Model Score: {model.score(X, y)}\n")

    return model
