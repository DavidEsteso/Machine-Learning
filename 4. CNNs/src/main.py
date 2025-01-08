import part1 as p1
import pandas as pd
import numpy as np
import part2 as p2


if __name__ == "__main__":

    kernel1 = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    kernel2 = np.array([
        [ 0, -1,  0],
        [-1,  8, -1],
        [ 0, -1,  0]
    ])
    
    p1.process_image('../data/shape.jpg', kernel1, "kernel1")
    p1.process_image('../data/shape.jpg', kernel2, "kernel2")

    # Train model with 5, 10, 20 and 40K samples
    p2.train_and_save_model("a_5k", n_samples=5000)
    p2.train_and_save_model("a_10k", n_samples=10000)
    p2.train_and_save_model("a_20k", n_samples=20000)
    p2.train_and_save_model("a_40k", n_samples=40000)

    # Train a model with pooling
    p2.train_and_save_model("b_pooling", n_samples=5000, pooling=True)



    # Train model with 5k samples and weight 0, 0.0001 ... 0.1
    p2.train_and_save_model("c_weight_0", n_samples=5000, weight=0.0)
    p2.train_and_save_model("c_weight_00001", n_samples=5000, weight=0.0001)
    p2.train_and_save_model("c_weight_001", n_samples=5000, weight=0.001)
    p2.train_and_save_model("c_weight_01", n_samples=5000, weight=0.01)
    p2.train_and_save_model("c_weight_1", n_samples=5000, weight=0.1)

    # Train a deeper model with 5, 10, 20 and 40K samples
    p2.train_and_save_model("d_5k", n_samples=5000, deeper=True)
    p2.train_and_save_model("d_10k", n_samples=10000, deeper=True)
    p2.train_and_save_model("d_20k", n_samples=20000, deeper=True)
    p2.train_and_save_model("d_40k", n_samples=40000, deeper=True)

    p2.train_logistic_regression_with_cv()





