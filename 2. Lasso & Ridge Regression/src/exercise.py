import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

def plot_3d(df):
    """
    Generate 3D and 2D scatter plots from a DataFrame with features and target.
    """
    X = df.iloc[:, :2].values 
    y = df.iloc[:, 2].values   
    
    # Normalize target values for color mapping
    norm = plt.Normalize(y.min(), y.max())
    colors = plt.cm.coolwarm(norm(y)) 
    
    # Create 3D scatter plot
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')  
    ax_3d.scatter(X[:, 0], X[:, 1], y, c=colors, marker='o')  
    ax_3d.set_xlabel('First feature (X1)')
    ax_3d.set_ylabel('Second feature (X2)') 
    ax_3d.set_zlabel('Target (y)') 
    
    # Add color bar for 3D plot
    color_map = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    color_map.set_array([])
    cbar = fig_3d.colorbar(color_map, ax=ax_3d, shrink=0.5, aspect=5)
    cbar.set_label('Target (y)')  

    
    plt.savefig('../output/i/3d_scatter_plot.png')


def train_model_with_polynomial_features(model_type, df, C_values=None):
    """
    Train a regression model with polynomial features (up to degree 5) 
    and varying C values for Lasso and Ridge regression.
    """
    
    X = df.iloc[:, :2].values
    y = df.iloc[:, 2].values
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    if model_type in ['lasso', 'ridge']:
        output_file = f'../output/i/{model_type}_results.txt'
        models = []
        
        with open(output_file, 'w') as f:
            for C in C_values:
                alpha = 1 / (2 * C)

                # Create and train the specified model
                if model_type == 'lasso':
                    model = Lasso(alpha=alpha)
                elif model_type == 'ridge':
                    model = Ridge(alpha=alpha)
                
                model.fit(X_poly, y)

                f.write(f"\n{model_type.capitalize()} with C = {C} (alpha = {alpha}):\n")
                f.write(f"Intercept: {model.intercept_}\n")
                f.write("Coefficients:\n")

                for feature, coef in zip(poly.get_feature_names_out(['X1', 'X2']), model.coef_):
                    f.write(f"{feature}: {coef}\n") 

                f.write(f"Model Score: {model.score(X_poly, y):.4f}\n")
                num_nonzero = np.sum(model.coef_ != 0)
                f.write(f"Non-zero coefficients: {num_nonzero}\n")
                f.write("-" * 50 + "\n")

                models.append(model)
        
        return models

    elif model_type == 'dummy_mean':
        output_file = '../output/i/dummy_mean_results.txt'
        
        # Train DummyRegressor with 'mean' strategy
        dummy = DummyRegressor(strategy='mean')
        dummy.fit(X_poly, y)

        y_pred = dummy.predict(X_poly)

        mse = mean_squared_error(y, y_pred)

        with open(output_file, 'w') as f:
            f.write("Dummy Regressor\n")
            f.write(f"Score: {dummy.score(X_poly, y):.4f}\n")
            f.write(f"Mean Squared Error: {mse:.4f}\n")
        
        return dummy


def plot_model_predictions(models, df):
    """
    Generate predictions from trained models and plot them with training data.
    Save each group of two models into separate files with different colors for each batch.
    """

    X = df.iloc[:, :2].values 
    y = df.iloc[:, 2].values 

    model_name = type(models[0]).__name__

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    # Create a grid of feature values for plotting
    padding = 2
    grid_x1 = np.linspace(x1_min - padding, x1_max + padding, 100)
    grid_x2 = np.linspace(x2_min - padding, x2_max + padding, 100)
    
    Xtest = []
    # Create a grid of feature values
    for i in grid_x1:
        for j in grid_x2:
            Xtest.append([i, j])
    Xtest = np.array(Xtest)

    norm = plt.Normalize(y.min(), y.max())
    colors = plt.cm.coolwarm(norm(y))

    custom_colors = ['red', 'yellow', 'green', 'blue']

    batch_size = 2

    for batch_idx in range(0, len(models), batch_size):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 3D scatter plot of training data
        ax.scatter(X[:, 0], X[:, 1], y, c=colors, marker='o', label='Training Data', alpha=0.5)

        ax.set_xlabel('First feature (X1)')
        ax.set_ylabel('Second feature (X2)')
        ax.set_zlabel('Target (y)')

        # Iterate through the current batch of models
        for idx, model in enumerate(models[batch_idx:batch_idx+batch_size]):
            # Calculate C from alpha
            if model_name != 'DummyRegressor':
                alpha = model.alpha
                C = 1 / (2 * alpha)

            poly = PolynomialFeatures(degree=5, include_bias=False)
            Xtest_poly = poly.fit_transform(Xtest)

            predictions = model.predict(Xtest_poly)

            predictions = predictions.reshape(100, 100)

            # Use custom colors for each model
            color = custom_colors[batch_idx + idx]

            # Plot the surface for each model
            if model_name == 'DummyRegressor':
                ax.plot_surface(grid_x1.reshape(100, 1), grid_x2.reshape(1, 100), predictions,
                                alpha=0.5, color='gray', label='Predictions (Dummy Mean)')
            else:
                ax.plot_surface(grid_x1.reshape(100, 1), grid_x2.reshape(1, 100), predictions,
                                alpha=0.3, color=color, label=f'Predictions (C={C})')

        # Set limits for the axes
        ax.set_xlim(X[:, 0].min() - padding, X[:, 0].max() + padding)
        ax.set_ylim(X[:, 1].min() - padding, X[:, 1].max() + padding)
        
        # Set z-limits to accommodate all models' predictions
        ax.set_zlim(y.min() - 15, y.max() + 15)

        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)

        ax.legend()

        plt.savefig(f'../output/i/{model_name.lower()}_predictions_batch_{batch_idx//batch_size + 1}.png')

        plt.close(fig)


def plot_prediction_error_vs_C(models, C_values, df):
    """
    Plot mean and standard deviation of prediction error (MSE) vs C values
    using 5-fold cross-validation for a list of models.
    """

    X = df.iloc[:, :2].values
    y = df.iloc[:, 2].values 

    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)

    model_name = type(models[0]).__name__

    output_file = f'../output/i/{model_name.lower()}_cross_validation.txt'

    mean_scores = []
    std_scores = []
    
    # Perform 5-fold cross-validation for each model
    for model in models:
        scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
        mean_scores.append(-scores.mean()) 
        std_scores.append(scores.std())  

    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)

    with open(output_file, 'w') as f:
        f.write("C values, Mean MSE, Std MSE\n")
        for C, mean, std in zip(C_values, mean_scores, std_scores):
            f.write(f"{C}, {mean}, {std}\n")
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(C_values, mean_scores, yerr=std_scores, fmt='o-', capsize=5, label='MSE ± STD')
    plt.xscale('log') 
    plt.xlabel('C (Regularization Strength)')
    plt.ylabel('Mean Cross-Validated MSE')
    plt.legend()
    plt.grid()
    plt.savefig(output_file.replace('.txt', '.png'))
