import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

def classify_iris_with_nn():
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    mlp = MLPClassifier(max_iter=1000)

    # Set up the hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(5,), (10,), (5, 5), (10, 10)],  # Number of layers and neurons in each layer
        'activation': ['logistic', 'tanh', 'relu'],              # Activation functions
        'solver': ['sgd', 'adam'],                               # Optimization algorithms
        'alpha': [0.0001, 0.001, 0.01],                         # L2 penalty (regularization term)
        'learning_rate': ['constant', 'adaptive']                # Learning rates
    }

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=3)
    grid_search.fit(X_train, y_train)

    # Best model after hyperparameter tuning
    best_model = grid_search.best_estimator_
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Test Accuracy: ", accuracy)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Call the function
classify_iris_with_nn()