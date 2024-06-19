import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Attributes to be used in the models
ATTRIBUTES = [
    "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", "w1", "w2", "w3", "wing_loading"
]

# Number of splits for cross-validation
N_SPLITS = 7


def clean_data(file_path: str) -> DataFrame:
    """
    Clean and preprocess the data from a CSV file, dropping rows with NaN values.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.

    Returns:
    - pandas.DataFrame: Cleaned and preprocessed DataFrame with NaN rows dropped.
    """
    df = pd.read_csv(file_path)

    # Replace '.' with NaN in 'Thorax_length' and 'wing_loading' columns
    df['Thorax_length'] = df['Thorax_length'].replace('.', np.nan)
    df['wing_loading'] = df['wing_loading'].replace('.', np.nan)

    # Convert 'Thorax_length' and 'wing_loading' columns to floats
    df['Thorax_length'] = df['Thorax_length'].astype(float)
    df['wing_loading'] = df['wing_loading'].astype(float)

    # Encode categorical columns
    species_mapping = {'D._aldrichi': 0, 'D._buzzatii': 1}
    df['Species'] = df['Species'].map(species_mapping)

    population_mapping = {'Gogango Creek': 0, 'Grandchester Wahruna': 1, 'Binjour': 2, 'Oxford Downs': 3}
    df['Population'] = df['Population'].map(population_mapping)

    sex_mapping = {'female': 0, 'male': 1}
    df['Sex'] = df['Sex'].map(sex_mapping)

    # Drop rows with NaN values
    df = df.dropna()

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix for the predicted results and print evaluation metrics.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    None
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"False Positive Rate: {round(fpr * 100, 2)}%")
    print(f"False Negative Rate: {round(fnr * 100, 2)}%")
    print(f"Accuracy: {round(accuracy * 100, 2)}%")


def evaluate_accuracy(y_true, y_pred):
    """
    Calculate and return the accuracy.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: Accuracy of the predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


def evaluate_false_negative_rate(y_true, y_pred):
    """
    Calculate and return the false-negative rate.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: False-negative rate of the predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp)
    return fnr


def evaluate_false_positive_rate(y_true, y_pred):
    """
    Calculate and return the false-positive rate.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: False-positive rate of the predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    return fpr


def xgboost(data_file_path):
    """
    Train and evaluate an XGBoost model.

    Parameters:
    data_file_path (str): Path to the CSV file containing the data.

    Returns:
    list: Model name, accuracy, false-negative rate, and false-positive rate.
    """
    df = clean_data(data_file_path)
    df_attributes = df[ATTRIBUTES]
    df_target = df["Species"]

    PARAM = {
        "max_depth": 4,
        "eta": 0.3359,
        "objective": "multi:softmax",
        "num_class": 2
    }
    EPOCHS = 500000
    EARLY_STOPPING_ROUNDS = 6000

    x_train, X_test, y_train, y_test = train_test_split(df_attributes, df_target, test_size=0.2)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)

    evals = [(test, "eval")]
    model = xgb.train(params=PARAM, dtrain=train, num_boost_round=EPOCHS, evals=evals,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False)

    predictions = model.predict(test)

    accuracy = evaluate_accuracy(y_test, predictions)
    fnr = evaluate_false_negative_rate(y_test, predictions)
    fpr = evaluate_false_positive_rate(y_test, predictions)

    return ["xg", accuracy, fnr, fpr]


def decision_tree(data_file_path):
    """
    Train and evaluate a Decision Tree model.

    Parameters:
    data_file_path (str): Path to the CSV file containing the data.

    Returns:
    list: Model name, accuracy, false-negative rate, and false-positive rate.
    """
    df = clean_data(data_file_path)
    df_attributes = df[ATTRIBUTES]
    df_target = df["Species"]

    x_train, x_test, y_train, y_test = train_test_split(df_attributes, df_target, test_size=0.2)

    model = DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = evaluate_accuracy(y_test, predictions)
    fnr = evaluate_false_negative_rate(y_test, predictions)
    fpr = evaluate_false_positive_rate(y_test, predictions)

    return ["dt", accuracy, fnr, fpr]


def neural_net(data_file_path):
    """
    Train and evaluate a Neural Network model.

    Parameters:
    data_file_path (str): Path to the CSV file containing the data.

    Returns:
    list: Model name, accuracy, false-negative rate, and false-positive rate.
    """
    df = clean_data(data_file_path)
    df_attributes = df[ATTRIBUTES]
    df_target = df["Species"]

    x_train, x_test, y_train, y_test = train_test_split(df_attributes, df_target, test_size=0.2)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = MLPClassifier(hidden_layer_sizes=(332, 191, 412), max_iter=100000, early_stopping=True)
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)

    accuracy = evaluate_accuracy(y_test, predictions)
    fnr = evaluate_false_negative_rate(y_test, predictions)
    fpr = evaluate_false_positive_rate(y_test, predictions)

    return ["nn", accuracy, fnr, fpr]


def threshold_predictions(y_prob, threshold):
    """
    Threshold the probabilities to get binary predictions.

    Parameters:
    y_prob (array-like): Predicted probabilities.
    threshold (float): Threshold value.

    Returns:
    list: Binary predictions based on the threshold.
    """
    return [1 if prob >= threshold else 0 for prob in y_prob]


if __name__ == "__main__":
    DATA_FILE_PATH = "data/83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv"

    xg = {"acc": [], "fnr": [], "fpr": []}
    dt = {"acc": [], "fnr": [], "fpr": []}
    nn = {"acc": [], "fnr": [], "fpr": []}

    for _ in range(200):
        name, acc, fnr, fpr = xgboost(DATA_FILE_PATH)
        xg["acc"].append(acc)
        xg["fnr"].append(fnr)
        xg["fpr"].append(fpr)

        name, acc, fnr, fpr = decision_tree(DATA_FILE_PATH)
        dt["acc"].append(acc)
        dt["fnr"].append(fnr)
        dt["fpr"].append(fpr)

        name, acc, fnr, fpr = neural_net(DATA_FILE_PATH)
        nn["acc"].append(acc)
        nn["fnr"].append(fnr)
        nn["fpr"].append(fpr)

    xg_df = pd.DataFrame.from_dict(xg)
    dt_df = pd.DataFrame.from_dict(dt)
    nn_df = pd.DataFrame.from_dict(nn)

    xg_95_acc = xg_df["acc"].quantile(0.95)
    dt_95_acc = dt_df["acc"].quantile(0.95)
    nn_95_acc = nn_df["acc"].quantile(0.95)

    print("xg 95th percentile accuracy:", xg_95_acc)
    print("dt 95th percentile accuracy:", dt_95_acc)
    print("nn 95th percentile accuracy:", nn_95_acc)
