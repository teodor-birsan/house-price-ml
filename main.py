import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle

def main():
    # Load the training dataset and validation data
    train_dataset = load_data("dataset\\train.csv")
    test_dataset = load_data("dataset\\test.csv")
    random_forrest_model(train_dataset, test_dataset)

    
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

def random_forrest_model(train_dataset: DataFrame, test_dataset: DataFrame):
    """Creates a random forrest model using train_dataset then fits the model, evaluates it, prints the mae
    and makes predictions with test_dataset

    Args:
        train_dataset (DataFrame): data used for training
        test_dataset (DataFrame): data used for validation
    """
    # The features that will affect the house prices
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    
    # Training data 
    y = train_dataset.SalePrice
    X = train_dataset[features]
    X = X.dropna(axis=0)
    print(X.describe())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # Validation data
    val_data = test_dataset[features]

    # Model: random forest with 100 trees
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train, y_train)
    predictions1 = model.predict(X_test)
    mae1 = mean_absolute_error(y_test, predictions1)

    predictions2 = model.predict(val_data)

    with open("log.txt", 'a') as f:
        f.write(f"{datetime.now()}:\n MAE for training data: {mae1} \n Prediction for new data: \n {predictions2} \n")

    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()