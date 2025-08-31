from datetime import datetime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

def prepare_data(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, features: list[str], target: list[str], test_size: float = 0.25):
    X = train_dataset[features]
    y = train_dataset[target] 
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=1)
    numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['float64', 'int64']]
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

    imputer = SimpleImputer(strategy='constant')
    encoder = OrdinalEncoder(encoded_missing_value=-2, handle_unknown='use_encoded_value', unknown_value=-1)

    transformer = ColumnTransformer(transformers=[
        ('num', imputer, numerical_cols),
        ('cat', encoder, categorical_cols)
    ])
    preprocessed_X_train = pd.DataFrame(transformer.fit_transform(X_train))
    preprocessed_X_valid = pd.DataFrame(transformer.transform(X_valid))
    preprocessed_X_train.index, preprocessed_X_train.columns = X_train.index, X_train.columns
    preprocessed_X_valid.index, preprocessed_X_valid.columns = X_valid.index, X_valid.columns
    preprocessed_training_data = (preprocessed_X_train, preprocessed_X_valid, y_train, y_valid)
    test_data = test_dataset[features]
    prerpocessed_test_data = pd.DataFrame(transformer.transform(test_data))
    prerpocessed_test_data.index, prerpocessed_test_data.columns = test_data.index, test_data.columns
    return (preprocessed_training_data, prerpocessed_test_data)


def create_output(predictions: list, index_list: list, save_to_file: bool = True, path: str = r"results\house_prices.csv") -> pd.DataFrame:
    output = pd.DataFrame({'Id': index_list, 'SalePrice': predictions})
    if save_to_file:
        output.to_csv(path, index=False)
    return output

def save_log(mae: float, nodes: int, log_path: str = r"results\log.txt", model_type: str = 'DEFAULT'):
    with open(log_path, "a") as f:
        f.write(f"{datetime.now()}, {model_type}:\n\tNodes: {nodes}\n\tMAE: {mae}\n")