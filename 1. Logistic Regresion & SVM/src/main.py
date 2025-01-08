import exercise_a as a
import exercise_b as b
import exercise_c as c


if __name__ == "__main__":
    # Load the data 
    data_filepath = '../data/week2.csv'
    df = a.load_data(data_filepath)

    # To solve exercise A
    a.visualize_data(df)
    model = a.train_logistic_regression(df)
    a.visualize_predictions_with_decision_boundary(df, model) 

    # To solve exercise B   
    C_values = [0.001, 0.01, 1, 10]  
    models = b.train_linear_svm(df, C_values)
    b.plot_svm_predictions(df, models, C_values)
    
    # To solve exercise C
    c.add_squared_features(df)
    df = a.load_data('../data/week2_squared.csv')
    model = a.train_logistic_regression(df)
    a.visualize_predictions_with_decision_boundary(df, model)  
    c.baseline_predictor(df)





