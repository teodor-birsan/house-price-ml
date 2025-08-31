import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def xgboost_model(dataset: pd.DataFrame):
    (X_train, X_valid, y_train, y_valid) = dataset[0]
    model = XGBRegressor(
        n_estimators=20000,
        early_stopping_rounds=10,
        learning_rate=0.1,
        n_jobs=4
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    nodes = len(model.get_booster().get_dump())
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    prediction = model.predict(dataset[1])
    return mae, nodes, prediction


