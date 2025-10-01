import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def classify_iris():
    # Load the iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = SVC()

    # Define hyperparameters to tune
    parameters = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # Setting up Grid Search
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print best parameters and best score
    print("Best Parameters:", best_params)
    print("Best Cross-Validation Score:", best_score)

    # Make predictions using the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate the model on the test set
    print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Call the function
classify_iris()