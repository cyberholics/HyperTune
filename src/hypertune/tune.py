from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Tune hyperparameters of a pre-trained model using GridSearchCV, print the results,
    and re-fit the model using the best parameters.

    Parameters:
    - model: A scikit-learn model (e.g., RandomForestClassifier).
    - param_grid: A dictionary of hyperparameters to search and their possible values.
    - X_train: The training feature dataset.
    - y_train: The training target variable.
    - cv: The number of cross-validation folds (default: 5).
    - scoring: The scoring metric to optimize (default: 'accuracy').

    Returns:
    - best_model: The best-trained model with optimized hyperparameters.
    """

    # GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    #Best model with optimised hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    #Results
    print("Best Model Parameters:")
    print(best_params)
    print(f"Best Score ({scoring}): {best_score:.2f}")

    # Re-fit the model using the best parameters
    best_model.fit(X_train, y_train)

    return best_model
