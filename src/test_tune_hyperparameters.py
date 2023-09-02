import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from hypertune.tune import tune_hyperparameters

# Sample data for testing
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def test_tune_hyperparameters():
    #RandomForestClassifier
    model = RandomForestClassifier(random_state=42)

    # Grid of hyperparameters to search
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
    }

    # tune_hyperparameters function
    best_model = tune_hyperparameters(model, param_grid, X_train, y_train, scoring='f1_macro')

    # Assertions to check if the function worked as expected
    assert isinstance(best_model, RandomForestClassifier)  # Check if the returned object is a RandomForestClassifier
    assert hasattr(best_model, 'n_estimators')  # Check if the best model has the 'n_estimators' attribute

    
