import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle


def load_data(path: str, describe: bool = False) -> DataFrame:
    """Loads a dataset from a csv file.

    Args:
        path (str): Path to the csv file.
        describe (bool, optional): Boolean value used to choose whether to print or not the description of the dataset. Defaults to False.

    Returns:
        DataFrame: the dataset from the csv file
    """
    dataset = pd.read_csv(path)
    if describe:
        print("Dataset details: \n", dataset.describe())
    return dataset

def random_forest(train_dataset: DataFrame, features: list[str], prediction_target: list[str], test_size: float = 0.3, nodes: int = 100, save: bool = False):
    X = train_dataset[features]
    y = train_dataset[prediction_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=test_size)

    model = RandomForestRegressor(
        random_state=1,
        n_estimators=nodes
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    if save:
        with open(f"models/random_forrest_n{nodes}.pkl", 'wb') as f:
            pickle.dump(model, f)
    return (model, mae)

def best_random_forest(train_dataset, validation_data, nodes_list, features: list[str],  test_size = 0.3, save: bool = False):
    best_nodes = nodes_list[0]
    best_model, min_mae = random_forest(train_dataset, nodes=nodes_list[0], test_size=test_size)
    for nodes in nodes_list:
        model, mae = random_forest(train_dataset=train_dataset, nodes=nodes, test_size=test_size)
        if mae < min_mae:
            best_nodes = nodes
            min_mae = mae
            best_model = model
    if save:
        with open(f"models/random_forrest_n{best_nodes}.pkl", 'wb') as f:
            pickle.dump(best_model, f)
    X_val = validation_data[features]
    predictions = best_model.predict(X_val)
    return (best_model, predictions, min_mae, best_nodes)
