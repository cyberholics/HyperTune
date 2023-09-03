# HyperTune

HyperTune is a Python package that uses [cross-validation](https://www.geeksforgeeks.org/cross-validation-machine-learning/) technique to help you find the best hyperparameter for your pre-trained model. It also helps you retrain your model with the best hyperparameters. It helps you automate the task of improving your model, giving you the opportunity to become productive.

## How to install HyperTune

## How to use HyperTune

To use Hypertune, make sure you:
- Have your pre-trained model
- Your training dataset
- Understand hyperparameter tuning in machine learning.
### Example of how use HyperTune to find the best hyperparameters for  a random forest model 
``` from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Import the modified tune_hyperparameters function
from hypertune.tune import tune_hyperparameters

# Load a dataset for demonstration (Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pre-trained model (RandomForestClassifier in this case)
model = RandomForestClassifier(random_state=42)

# Define a grid of hyperparameters to search
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
}
 
# Use the tune_hyperparameters function to tune hyperparameters and obtain the best model
best_model = tune_hyperparameters(model, param_grid, X_train, y_train, scoring='f1_macro')

# The best_model is now the RandomForestClassifier with optimized hyperparameters
# You can use it for predictions 
y_pred = best_model.predict(X_test) ```

