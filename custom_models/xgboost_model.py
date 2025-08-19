import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import configparser

def xgboost_model(features: list[str], prediction_target: list[str], save_to_csv: bool = False, new_data: str = None, index: int = None):
    dataset = pd.read_csv("dataset\\train.csv", index_col=0)
    X = dataset[features]
    y = dataset[prediction_target]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=2)
    numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['float64', 'int64']]
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
    
    numerical_imputer = SimpleImputer(strategy='constant')
    categorical_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(
            #categories=order_categories("categories_order.ini"), 
            encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-2))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_imputer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    preprocessed_X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
    preprocessed_X_valid = pd.DataFrame(preprocessor.fit_transform(X_valid))
    preprocessed_X_train.index = X_train.index
    preprocessed_X_train.columns = X_train.columns
    preprocessed_X_valid.index = X_valid.index
    preprocessed_X_valid.columns = X_valid.columns
    model = XGBRegressor(n_estimators=20000, early_stopping_rounds=10, learning_rate=0.05, n_jobs=4)
    model.fit(preprocessed_X_train, y_train,
              eval_set=[(preprocessed_X_valid, y_valid)],
              verbose=False)
    predictions = model.predict(preprocessed_X_valid)
    mae = mean_absolute_error(y_valid, predictions)
    rmse = root_mean_squared_error(y_valid, predictions)
    print(f"MAE: {mae}\nRMSE: {rmse}\n")

    if save_to_csv:
        try:
            test_data = pd.read_csv(new_data, index_col=index)
            test_data = test_data[features]
            preprocessed_test_data = pd.DataFrame(preprocessor.transform(test_data))
            preprocessed_test_data.columns = test_data.columns
            preprocessed_test_data.index = test_data.index
            prediction = model.predict(preprocessed_test_data)
            output = pd.DataFrame({'Id': test_data.index, 'SalePrice': prediction})
            output.to_csv("results\\xgboost_house_prices.csv", index=False)
        except Exception as e:
            print(e)


def order_categories(path: str):
    config = configparser.ConfigParser()
    config.read(path)
    categories = config['Categories']
    ordered_categories = []
    for val in categories.values():
        ordered_categories.append(val.split(' '))
    return ordered_categories
            