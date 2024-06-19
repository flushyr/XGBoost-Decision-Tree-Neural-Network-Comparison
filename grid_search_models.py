import time

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from models import clean_data, ATTRIBUTES


# Function to perform grid search for XGBoost model
def grid_search_xgboost(data_file_path):
    print("Grid Search XGBoost")
    start_time = time.time()

    # Load and preprocess data
    df = clean_data(data_file_path)
    df_attributes = df[ATTRIBUTES]
    df_target = df["Species"]

    # Define parameter grid for grid search
    PARAM_GRID = {
        "max_depth": [3, 4, 5, 6, 7],
        "eta": [0.1, 0.2, 0.3, 0.4, 0.5],
        "num_class": [2]
    }

    # Split data into training and testing sets
    x_train, X_test, y_train, y_test = train_test_split(df_attributes, df_target, test_size=0.2, random_state=1425)

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(objective="multi:softmax", random_state=1425)

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=PARAM_GRID, cv=3)
    grid_search.fit(x_train, y_train)

    # Get best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions and calculate accuracy
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    end_time = time.time()
    time_taken = "{:.0f}".format((end_time - start_time) * 1000)  # Time taken in milliseconds

    # Print results
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken: {time_taken} milliseconds")


# Function to perform grid search for Decision Tree model
def grid_search_decision_tree(data_file_path):
    print("Grid Search Decision Tree")
    start_time = time.time()

    # Load and preprocess data
    df = clean_data(data_file_path)
    df_attributes = df[ATTRIBUTES]
    df_target = df["Species"]

    # Define parameter grid for grid search
    PARAM_GRID = {
        "max_depth": [5, 10, 15, 20, 25, 30],
        "min_samples_split": [2, 5, 10, 15, 20, 25],
        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"]
    }

    # Split data into training and testing sets
    x_train, X_test, y_train, y_test = train_test_split(df_attributes, df_target, test_size=0.2, random_state=1425)

    # Initialize Decision Tree classifier
    model = DecisionTreeClassifier(random_state=1425)

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=PARAM_GRID, cv=3)
    grid_search.fit(x_train, y_train)

    # Get best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions and calculate accuracy
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    end_time = time.time()
    time_taken = "{:.0f}".format((end_time - start_time) * 1000)  # Time taken in milliseconds

    # Print results
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken: {time_taken} milliseconds")


# Function to perform grid search for Neural Network model
def grid_search_neural_net(data_file_path):
    print("Grid Search Neural Network")
    start_time = time.time()

    # Load and preprocess data
    df = clean_data(data_file_path)
    df_attributes = df[ATTRIBUTES]
    df_target = df["Species"]

    # Define parameter grid for grid search
    PARAM_GRID = {
        "hidden_layer_sizes": [(100, 50), (200, 100), (300, 150), (400, 200), (500, 250)],
        "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    }

    # Split data into training and testing sets
    x_train, X_test, y_train, y_test = train_test_split(df_attributes, df_target, test_size=0.2, random_state=1425)

    # Standardize data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(X_test)

    # Initialize Neural Network classifier
    model = MLPClassifier(activation='relu', solver='adam', max_iter=100000, early_stopping=True, random_state=1425)

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=PARAM_GRID, cv=3)
    grid_search.fit(x_train_scaled, y_train)

    # Get best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions and calculate accuracy
    predictions = best_model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    end_time = time.time()
    time_taken = "{:.0f}".format((end_time - start_time) * 1000)  # Time taken in milliseconds

    # Print results
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken: {time_taken} milliseconds")


if __name__ == "__main__":
    # Path to the dataset
    data_file_path = "data/83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv"

    # Perform grid searches and print results
    grid_search_xgboost(data_file_path)
    print()
    grid_search_decision_tree(data_file_path)
    print()
    grid_search_neural_net(data_file_path)
