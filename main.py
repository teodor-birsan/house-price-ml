from data.data import create_output, prepare_data, save_log
import pandas as pd
from models.basic.random_forest import best_random_forest
from models.basic.xgboost_model import xgboost_model


HOUSE_FEATURES = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                  'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 
                  'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 
                  'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 
                  'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
                  '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                  'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
                  'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 
                  'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 
                  'MiscVal', 'SaleType', 'SaleCondition']
TARGET_PREDICTION = "SalePrice"


def main():
    training_dataset = pd.read_csv(r"dataset\train.csv", index_col=0)
    test_dataset = pd.read_csv(r"dataset\test.csv", index_col=0)
    dataset = prepare_data(
        train_dataset=training_dataset,
        test_dataset=test_dataset,
        features=HOUSE_FEATURES,
        target=TARGET_PREDICTION
    )
    mae, estimators, prediction = xgboost_model(dataset)
    save_log(mae=mae, nodes=estimators, model_type='XGBoost Model')
    create_output(predictions=prediction, index_list=test_dataset.index, path=r"results\house_prices_xgboost.csv")

if __name__ == "__main__":
    main()