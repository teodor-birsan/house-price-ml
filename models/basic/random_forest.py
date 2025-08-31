from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

def best_random_forest(dataset: tuple[pd.DataFrame], nodes_list: list[int]):
    training_data = dataset[0]
    (X_train, X_valid, y_train, y_valid) = training_data
    best_model, min_mae = create_random_forest(nodes_list[0], X_train, y_train, X_valid, y_valid)
    best_nodes = nodes_list[0]
    for nodes in nodes_list[1:]:
        model, mae = create_random_forest(nodes, X_train, y_train, X_valid, y_valid)
        if mae < min_mae:
            best_model, min_mae = model, mae
            best_nodes = nodes
    test_data = dataset[1]
    prediction = best_model.predict(test_data)
    return (min_mae, best_nodes, prediction)


def create_random_forest(nodes: int, X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    rf_model = RandomForestRegressor(n_estimators=nodes, random_state=1)
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_valid)
    mae = mean_absolute_error(y_valid, predictions)
    return (rf_model, mae)