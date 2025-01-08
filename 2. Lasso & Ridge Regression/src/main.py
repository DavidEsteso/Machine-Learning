import exercise as i
import pandas as pd


if __name__ == "__main__":
    data_filepath = '../data/week3.csv'
    df = pd.read_csv(data_filepath)
    i.plot_3d(df)

    dummy = i.train_model_with_polynomial_features('dummy_mean', df)
    i.plot_model_predictions([dummy], df)

    C_values_lasso = [1, 10, 100, 1000]
    lasso = i.train_model_with_polynomial_features('lasso', df, C_values_lasso)
    i.plot_model_predictions(lasso, df)

    C_values_ridge = [0.01, 0.1, 1, 10]
    ridge = i.train_model_with_polynomial_features('ridge', df, C_values_ridge)
    i.plot_model_predictions(ridge, df)

    i.plot_prediction_error_vs_C(ridge,  C_values_ridge,df)
    i.plot_prediction_error_vs_C(lasso, C_values_lasso, df)





