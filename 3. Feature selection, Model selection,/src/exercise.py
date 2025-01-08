import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.dummy import DummyClassifier


def load_data(filepath):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def cross_val_nested_loops(df, max_poly_order_range, C_range, dataset):
    """Train and evaluate a Logistic Regression model using nested loops. Return the best model, best polynomial features, and results."""
    X = df.iloc[:, :2].values  
    y = df.iloc[:, 2].values   
    
    best_score = 0
    results = []
    best_model = None
    best_poly = None

    for max_poly_order in max_poly_order_range:
        poly = PolynomialFeatures(degree=max_poly_order)
        X_poly = poly.fit_transform(X)

        for C in C_range:
            model = LogisticRegression(C=C, penalty='l2', max_iter=100000)
            scores = cross_val_score(model, X_poly, y, cv=5, scoring='accuracy')
            mean_score = scores.mean()
            std_score = scores.std()

            results.append((max_poly_order, C, mean_score, std_score))
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_poly = poly
    
    # Save the best results to a file
    results_filename = f"../output/{dataset}/best_results_lr.txt"
    with open(results_filename, 'w') as file:
        file.write(f"Best Polynomial Order: {best_poly.degree}\n")
        file.write(f"Best  C: {best_model.C}\n")
        file.write(f"Best Cross-Validation Score: {best_score:.4f}\n")

    return best_model, best_poly, results



def plot_cross_val_results(results, name, dataset):
    """Plot cross-validation results and save to a file."""
    if name == 'lr':
        results_df = pd.DataFrame(results, columns=['Max Poly Order', 'C', 'Mean Score', 'Std Score'])
    elif name == 'knn':
        results_df = pd.DataFrame(results, columns=['K', 'Mean Score', 'Std Score'])

    plt.figure(figsize=(12, 6))
    
    # Handle plotting differently based on the model name
    if name == 'lr':
        for max_order in results_df['Max Poly Order'].unique():
            subset = results_df[results_df['Max Poly Order'] == max_order]
            plt.errorbar(subset['C'], subset['Mean Score'], yerr=subset['Std Score'], 
                         label=f'Poly Order: {max_order}', marker='o')
            plt.xscale('log')  

        plt.xlabel('C')
        
    elif name == 'knn':
        plt.errorbar(results_df['K'], results_df['Mean Score'], yerr=results_df['Std Score'], 
                     label='KNN', marker='o')
        plt.xlabel('Number of Neighbors (K)')
    
    plt.ylabel('Mean Accuracy')
    plt.legend()
    plt.grid()
    
    filename = f"../output/{dataset}/cv_{name}.png"
    plt.savefig(filename)  
    plt.close()  


def plot_and_save_results(df, model, poly, name, dataset):
    """Visualize training data and predictions of the model, and save the confusion matrix and metrics. Retunr the true labels and predictions."""  
    features = df.shape[1] - 1 
    X = df.iloc[:, :features].values  
    y = df.iloc[:, features].values    
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform the features if polynomial features are provided
    X_train_poly = poly.fit_transform(X_train) if poly is not None else X_train
    X_test_poly = poly.fit_transform(X_test) if poly is not None else X_test

    model.fit(X_train_poly, y_train)  

    predictions = model.predict(X_test_poly)  

    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], marker='+', color='purple', label='Training +1')
    plt.scatter(X_train[:, 0][y_train == -1], X_train[:, 1][y_train == -1], marker='o', color='orange', label='Training -1')
    
    # Plot predictions on the test set
    plt.scatter(X_test[:, 0][predictions == 1], X_test[:, 1][predictions == 1], marker='x', color='pink', label='Predicted +1')
    plt.scatter(X_test[:, 0][predictions == -1], X_test[:, 1][predictions == -1], marker='x', color='yellow', label='Predicted -1')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.legend(loc='lower right')
    plt.grid(True)

    filename = f"../output/{dataset}/predictions_{name}.png"
    plt.savefig(filename)
    plt.close()  

    # Create a confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=[-1, 1])

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(y_test, predictions, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_test, predictions, average='binary', pos_label=1, zero_division=0)


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 1])
    plt.figure(figsize=(8, 6))

    disp.plot(cmap='Blues', values_format='d')
    
    # Save the confusion matrix plot to a file
    cm_filename = f"../output/{dataset}/{name}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()  

    # Save the confusion matrix and metrics to a text file
    metrics_filename = f"../output/{dataset}/{name}_metrics.txt"
    with open(metrics_filename, 'w') as f:
        f.write("Confusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')

        f.write("\nMetrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")


    # Save LR information to a file
    if name == 'lr':
        filename = f"../output/{dataset}/logistic_regression_model_info.txt"
        with open(filename, 'w') as file:
            file.write("intercept: " + str(model.intercept_[0]) + "\n")
            for feature, coef in zip(poly.get_feature_names_out(['X1', 'X2']), model.coef_):
                file.write(f"{feature}: {coef}\n")
    return y_test, predictions


def train_knn_with_cross_validation(df , k_values, dataset):
    """Train a kNN classifier on the data and use cross-validation to select the best k and return the model and results."""

    features = df.shape[1] - 1 
    X = df.iloc[:, :features].values  
    y = df.iloc[:, features].values   
    results = [] 
    best_k = 0
    best_score = 0

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results.append((k, mean_score, std_score))

    for k, score, _ in results:
        if score > best_score:
            best_k = k
            best_score = score


    # Train the final model with the best k on the full dataset
    final_knn = KNeighborsClassifier(n_neighbors=best_k)

    # Save the best results to a file
    results_filename = f"../output/{dataset}/best_results_knn.txt"
    with open(results_filename, 'w') as file:
        file.write(f"Best Number of Neighbors (k): {best_k}\n")
        file.write(f"Best Cross-Validation Score: {best_score:.4f}\n")

    return final_knn, results 


def train_dummy_classifier(df):
    """Train a DummyClassifier on the provided DataFrame and return the model"""
    X = df.iloc[:, :2].values  
    y = df.iloc[:, 2].values   


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the DummyClassifier with the 'most_frequent' strategy
    dummy_classifier = DummyClassifier(strategy='most_frequent') 
    dummy_classifier.fit(X_train, y_train)


    return dummy_classifier


def plot_roc_curve(y_tests, predictions, names, dataset):
    """Plot ROC curves for the models and save the plot to a file."""
    plt.figure(figsize=(10, 6))

    # predictions and y_tests are lists of model predictions and true labels
    for i in range(len(predictions)):
        fpr, tpr, _ = roc_curve(y_tests[i], predictions[i])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve for each model
        plt.plot(fpr, tpr, label=f'{names[i]} (AUC = {roc_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f"../output/{dataset}/roc_curves.png")
    plt.close()  