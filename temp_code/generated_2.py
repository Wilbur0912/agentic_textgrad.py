import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def classify_iris():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    # Evaluate each model
    best_model = None
    best_accuracy = 0
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f'{model_name} Accuracy: {accuracy:.4f}')
        
        # Update the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name

    print(f'Best model: {best_model} with accuracy {best_accuracy:.4f}')

    return best_model, best_accuracy

# Run the function to classify iris species
best_model, best_accuracy = classify_iris()