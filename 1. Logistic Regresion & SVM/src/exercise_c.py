import pandas as pd
import exercise_a as a

def add_squared_features(df, output_file='../data/week2_squared.csv'):
    """Add squared features to the DataFrame and save to a CSV file."""
    X1 = df.iloc[:, 0]
    X2 = df.iloc[:, 1]
    
    #Generate squared features
    X1_squared = X1 ** 2
    X2_squared = X2 ** 2
    
    #Create a new feature DataFrame
    squared_features_df = pd.DataFrame({
        'Feature 1': X1,
        'Feature 2': X2,
        'Feature 1 Squared': X1_squared,
        'Feature 2 Squared': X2_squared,
        'Target': df.iloc[:, 2] 
    })
    
    squared_features_df.to_csv(output_file, index=False)

def baseline_predictor(df, output_file='../output/c/exercise_iii.txt'):
    """Train a baseline model that always predicts the most frequent class."""
    target = df.iloc[:, -1]
    
    most_frequent_class = target.mode()[0]
    most_frequent_count = target.value_counts()[most_frequent_class]
    
    accuracy_score = most_frequent_count / len(target)
    
    with open(output_file, 'w') as f:
        f.write(f'Accuracy Score: {accuracy_score:.4f}\n')
