import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_linear_svm(df, C_values):
    """Train linear SVM classifiers on data from a DataFrame for a range of C values."""
    X = df.iloc[:, :2].values
    y = df.iloc[:, 2].values
    models = []

    # Train a linear SVM model for each value of C
    with open('../output/b/exercise_i.txt', 'w') as f:
        for C in C_values:
            # Train a linear SVM model
            model = LinearSVC(C=C)
            model.fit(X, y)
            models.append(model)

            coefficients = model.coef_[0]
            intercept = model.intercept_[0]

            f.write(f"SVM Model with C={C}:\n")
            f.write(f"Coefficients: {coefficients}\n")
            f.write(f"Intercept: {intercept}\n\n")
            f.write(f"Number of Iterations: {model.n_iter_}\n")
            f.write(f"Model Score: {model.score(X, y)}\n")
            f.write(f"Regularization (C): {C}\n")
            f.write(f"\n\n")

    return models

def plot_svm_predictions(df, models, C_values):
    """Plot actual vs predicted values with decision boundaries for multiple SVM models."""
    X = df.iloc[:, :2].values
    y = df.iloc[:, 2].values

    for i, model in enumerate(models):
        # Make predictions using the model
        predictions = model.predict(X)
        
        x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
        y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        
        c = model.intercept_[0]
        m = model.coef_[0]
        x_values = np.linspace(x_min, x_max, 100)
        # Solve for y values using the decision boundary equation: c + m1*x + m2*y = 0
        y_values = -(c + m[0] * x_values) / m[1]

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, color='black', label='Decision Boundary')
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=20)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='purple', label='Actual +1')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='orange', label='Actual -1')
        plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], marker='x', color='pink', label='Predicted +1')
        plt.scatter(X[predictions == -1, 0], X[predictions == -1, 1], marker='x', color='yellow', label='Predicted -1')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.legend(loc='best')
        plt.grid(True)

        # Save each plot in a separate file
        plt.savefig(f'../output/b/exercise_ii_C_{C_values[i]}.png')
        plt.close() 
