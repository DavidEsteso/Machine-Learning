import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Flatten, Conv2D
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import time  
from sklearn.preprocessing import StandardScaler  

def define_model(input_shape, num_classes, deeper=False, pooling=False, weight=0.0001):
    model = keras.Sequential()
    
    if deeper:
        model.add(Conv2D(8, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    else:
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        if pooling:
            model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # Capa de max-pooling
        else:
            model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        if pooling:
            model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))  
        else:
            model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))

    # Add dropout, flatten and dense layer
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(weight)))

    return model

def train_and_save_model(id, num_classes=10, input_shape=(32, 32, 3), n_samples=5000, batch_size=128, epochs=20, weight=0.0001, pooling=False, deeper=False):
    output_dir = f"../output/{id}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train[:n_samples]
    y_train = y_train[:n_samples]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("Original x_train shape:", x_train.shape)

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Define the model
    model = define_model(input_shape, num_classes, deeper=deeper, pooling=pooling, weight=weight)

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()

    # Save the model summary
    with open(os.path.join(output_dir, "model_summary.txt"), "w", encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Measure the execution time
    start_time = time.time()

    # Train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save(os.path.join(output_dir, "cifar_model.h5"))

    # Stop time after training
    execution_time = time.time() - start_time

    # Plot accuracy and loss
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Model Accuracy', fontsize=16)  
    plt.ylabel('Accuracy', fontsize=16)     
    plt.xlabel('Epoch', fontsize=16)        
    plt.legend(loc='upper left', fontsize=14) 

    # Plot loss
    plt.subplot(212)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Model Loss', fontsize=16)   
    plt.ylabel('Loss', fontsize=16)       
    plt.xlabel('Epoch', fontsize=16)      
    plt.legend(loc='upper left', fontsize=14)   
    # Adjust spacing between the plots
    plt.subplots_adjust(hspace=0.6)  

    plt.savefig(os.path.join(output_dir, "model_performance.png"))
    plt.close()
    # Predictions and metrics
    evaluate_model(model, x_train, y_train, x_test, y_test, output_dir, execution_time)


def evaluate_model(model, x_train, y_train, x_test, y_test, output_dir, execution_time):
    # Predictions and metrics for training data
    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)

    # Save classification report and confusion matrix for training data
    with open(os.path.join(output_dir, "classification_report_train.txt"), "w", encoding='utf-8') as f:
        f.write(classification_report(y_train1, y_pred))

    cm_train = confusion_matrix(y_train1, y_pred)
    cm_train_rounded = np.round(cm_train, 3) 

    with open(os.path.join(output_dir, "confusion_matrix_train.txt"), "w", encoding='utf-8') as f:
        np.savetxt(f, cm_train_rounded, fmt='%.3f', delimiter=',', header="Predicted Class,0,1,2,3,4,5,6,7,8,9", comments="")

    # Predictions and metrics for test data
    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)

    # Save classification report and confusion matrix for test data
    with open(os.path.join(output_dir, "classification_report_test.txt"), "w", encoding='utf-8') as f:
        f.write(classification_report(y_test1, y_pred))

    cm_test = confusion_matrix(y_test1, y_pred)
    cm_test_rounded = np.round(cm_test, 3)

    with open(os.path.join(output_dir, "confusion_matrix_test.txt"), "w", encoding='utf-8') as f:
        np.savetxt(f, cm_test_rounded, fmt='%.3f', delimiter=',', header="Predicted Class,0,1,2,3,4,5,6,7,8,9", comments="")

    # Analyze model parameters
    total_params = model.count_params()
    layer_params = {layer.name: layer.count_params() for layer in model.layers}
    max_layer = max(layer_params, key=layer_params.get)
    max_params = layer_params[max_layer]
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)

    common_label = np.argmax(np.bincount(y_test1))
    baseline_accuracy = np.sum(y_test1 == common_label) / len(y_test1)

    results = (
        f"Total parameters in the model: {total_params}\n"
        f"Layer with most parameters: {max_layer} ({max_params} parameters)\n"
        f"Test accuracy: {test_accuracy * 100:.2f}%\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Train accuracy: {train_accuracy * 100:.2f}%\n"
        f"Train loss: {train_loss:.4f}\n"
        f"Execution time: {execution_time:.2f} seconds\n"
        f"Baseline accuracy (most common label): {baseline_accuracy * 100:.2f}%\n"
    )

    # Save the analysis results to a text file
    with open(os.path.join(output_dir, "analysis.txt"), "w", encoding='utf-8') as f:
        f.write(results)

def train_logistic_regression_with_cv(n_samples=5000):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Use only the first n_samples
    x_train = x_train[:n_samples]
    y_train = y_train[:n_samples]
    
    # Flatten the images
    x_train = x_train.reshape(x_train.shape[0], -1)  
    x_test = x_test.reshape(x_test.shape[0], -1)   
    
    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create a logistic regression model
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

    # Perform cross-validation
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')

    # Print the mean and standard deviation of the cross-validation scores
    with open("../output/cross_validation_results.txt", "w", encoding='utf-8') as f:
        f.write(f'Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}\n')

    model.fit(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    with open("../output/cross_validation_results.txt", "a", encoding='utf-8') as f:
        f.write(f'Test Accuracy: {test_accuracy:.4f}\n')