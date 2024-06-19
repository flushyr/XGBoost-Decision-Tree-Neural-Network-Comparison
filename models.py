import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Global Constants
ATTRIBUTES = [
    "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", "w1", "w2", "w3", "wing_loading"
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS = 5


# Function to clean and preprocess data
def clean_data(file_path: str) -> DataFrame:
    """
    Clean and preprocess the data from a CSV file, dropping rows with NaN values.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.

    Returns:
    - pandas.DataFrame: Cleaned and preprocessed DataFrame with NaN rows dropped.
    """
    # Clean and preprocess the data
    df = pd.read_csv(file_path)
    df['Thorax_length'] = df['Thorax_length'].replace('.', np.nan)
    df['wing_loading'] = df['wing_loading'].replace('.', np.nan)
    df['Thorax_length'] = df['Thorax_length'].astype(float)
    df['wing_loading'] = df['wing_loading'].astype(float)
    species_mapping = {'D._aldrichi': 0, 'D._buzzatii': 1}
    df['Species'] = df['Species'].map(species_mapping)
    sex_mapping = {'female': 0, 'male': 1}
    df['Sex'] = df['Sex'].map(sex_mapping)
    df = pd.get_dummies(df, columns=['Population'], prefix='population_is')
    df = df.dropna()
    return df


# Function to plot correlation matrix
def plot_correlation_matrix(file_path: str, attributes: list, figsize=(10, 8), cmap="flare"):
    """
    Load data from a CSV file, calculate the correlation matrix for specified attributes,
    and plot the correlation matrix as a heatmap.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.
    - attributes (list): List of attributes to include in the correlation matrix.
    - figsize (tuple, optional): Figure size for the heatmap plot. Default is (10, 8).
    - cmap (str, optional): Color map for the heatmap. Default is 'flare'.

    Returns:
    - None
    """
    df = clean_data(file_path)
    correlation_matrix = df[attributes].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix', fontname='Arial', fontsize=24)
    plt.xticks(rotation=45, fontname='Arial', fontsize=18)
    plt.yticks(rotation=0, fontname='Arial', fontsize=18)
    plt.tight_layout()
    plt.show()


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix for the predicted results.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    None
    """
    # Calculate confusion matrix and performance metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"False Positive Rate: {round(fpr * 100, 2)}%")
    print(f"False Negative Rate: {round(fnr * 100, 2)}%")
    print(f"Accuracy: {round(accuracy * 100, 2)}%")


# Function to train an XGBoost model with grid search
def xgboost(data_file_path):
    """
    Train an XGBoost model using grid search and evaluate its performance.

    Parameters:
    - data_file_path (str): Path to the CSV file containing the data.

    Returns:
    - None
    """
    df = clean_data(data_file_path)
    X = df.drop('Species', axis=1)
    y = df['Species']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.001]
    }

    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=N_SPLITS)
    grid_search.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = grid_search.best_estimator_.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)


# Function to train a neural network model
def neural_network(data_file_path):
    """
    Train a neural network model using PyTorch and evaluate its performance.

    Parameters:
    - data_file_path (str): Path to the CSV file containing the data.

    Returns:
    - None
    """
    df = clean_data(data_file_path)
    X = df.drop('Species', axis=1)
    y = df['Species']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.int64)

    # Define neural network architecture
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)  # Output layer with 2 classes (binary classification)
    ).to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    epochs = 100
    batch_size = 64
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate model performance
    model.eval()
    y_pred_tensor = model(X_test_tensor.to(DEVICE))
    y_pred = torch.argmax(y_pred_tensor, axis=1).cpu().numpy()
    plot_confusion_matrix(y_test, y_pred)


# Main function
if __name__ == "__main__":
    file_path = "Drosophila/drosophila_data.csv"

    # Plot correlation matrix
    plot_correlation_matrix(file_path, ATTRIBUTES)

    # Train and evaluate XGBoost model
    xgboost(file_path)

    # Train and evaluate neural network model
    neural_network(file_path)
