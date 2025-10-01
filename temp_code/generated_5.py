import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def load_and_classifier_iris():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Define the grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }

    # Set up the GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    
    # Fit the model
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print("Best Hyperparameters: ", grid_search.best_params_)

    # Evaluate on the test set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    print("Accuracy on test set: ", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Run the function
load_and_classifier_iris()