import exercise as e
import pandas as pd

if __name__ == "__main__":
    for i in range(1, 3):
        data_filepath = f'../data/data{i}.csv'
        df = pd.read_csv(data_filepath)


        model_lr, poly, results = e.cross_val_nested_loops(df, range(1, 10), [0.01, 0.1, 1, 10, 100, 1000], i)
        e.plot_cross_val_results(results, 'lr', i)
        test_lr, predictions_lr = e.plot_and_save_results(df, model_lr, poly, 'lr', i)


        model_knn, results = e.train_knn_with_cross_validation(df, range(1, 50), i)
        e.plot_cross_val_results(results, 'knn', i)
        test_knn, predictions_knn = e.plot_and_save_results(df, model_knn, None, 'knn', i)


        model_dummy = e.train_dummy_classifier(df)
        test_dummy, predictions_dummy= e.plot_and_save_results(df, model_dummy, None, 'dummy', i)

        e.plot_roc_curve(
            [test_lr, test_knn, test_dummy], 
            [predictions_lr, predictions_knn, predictions_dummy], 
            ['Logistic Regression', 'K-Nearest Neighbors', 'Dummy Classifier'], 
            i
        )




