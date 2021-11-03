"""
Script to perform tests by use of the pytest suite
"""
import os
from os import path
import logging
import pandas as pd
import joblib
import churn_library as cl
import constants


def test_import(test_dataframe):
    """
    tests data import - Uses the test_dataframe fixture to read in the csv
    """
    try:
        assert test_dataframe.shape[0] > 0
        assert test_dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows or columns")
        raise err


def test_create_churn_col():
    """
    tests create_churn_col
    """
    dataframe = pd.DataFrame(columns=['Attrition_Flag'])
    dataframe.loc[0] = ["Existing Customer"]
    dataframe.loc[1] = ["Attrited Customer"]
    dataframe = cl.create_churn_col(dataframe)
    try:
        assert dataframe.Churn.loc[0] == 0
        assert dataframe.Churn.loc[1] == 1
        logging.info(
            "Testing create_churn_col: The churn column was created successfully: SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing create_churn_col: The transformation from Attrition_Flag to Churn bool "
            "was not successful")
        raise err


def test_eda(test_dataframe_w_churn, tmpdir):
    """
    tests perform eda function. Use a fixture to read a test dataset and check if plot files
     are produced.
    """

    figures = [
        'Churn',
        'Customer_Age',
        'Marital_Status',
        'Total_Trans_Ct',
        'Correlations']

    try:
        cl.perform_eda(test_dataframe_w_churn, tmpdir)
        for figure in figures:
            assert path.isfile(
                path.join(
                    tmpdir,
                    'images',
                    'eda',
                    figure +
                    '.pdf'))
        logging.info(
            "Testing test_eda: All 5 eda figures created: SUCCESS"
        )
    except FileNotFoundError as err:
        logging.error(
            "Testing test_eda: Not all 5 eda figures were created"
        )
        raise err


def test_encoder_helper(test_dataframe_encoded):
    """
    tests encoder helper
    """
    try:
        for column in constants.CATEGORY_LST:
            assert column + '_' + constants.CHURN in test_dataframe_encoded.columns
        logging.info(
            'Testing test_encoder_helper: All expected categorical columns encoded: SUCCESS'
        )
    except AssertionError as err:
        logging.info(
            "Testing test_encoder_helper: At least one of the categorical columns was not created")
        raise err


def test_perform_feature_engineering(features):
    """
    tests perform_feature_engineering
    """
    try:
        x_train = features[0]
        x_test = features[1]
        y_train = features[2]
        y_test = features[3]
        assert len(x_train) == len(y_train) != 0
        assert len(x_test) == len(y_test) != 0
        logging.info(
            "Testing test_perform_feature_engineering: Features created: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_perform_feature_engineering: Features length mismatch or empty")
        raise err


def test_train_models(features, tmpdir):
    """
    tests train_models
    """
    if not path.exists(path.join(tmpdir, 'images', 'results')):
        os.makedirs(path.join(tmpdir, 'images', 'results'))
    cl.train_models(
        features[0],
        features[1],
        features[2],
        features[3],
        tmpdir
    )
    try:
        joblib.load(path.join(tmpdir, 'models', 'rfc_model' + '.pkl'))
        joblib.load(path.join(tmpdir, 'models', 'logistic_model' + '.pkl'))
        logging.info(
            "Testing train_models: Models created and stored: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Not all models were stored")
        raise err

    for image_name in [
        "Random_Forest",
        "Logistic_Regression",
        "Feature_Importance",
            'Prediction_Explainer']:
        try:
            assert path.isfile(
                path.join(
                    tmpdir,
                    'images',
                    'results',
                    image_name +
                    '.jpg')
            )
            logging.info(
                    "Testing train_models: Creating results images: SUCCESS"
            )
        except FileNotFoundError as err:
            logging.error(
                "Testing train_models: Generated results images missing")
            raise err


if __name__ == "__main__":
    pass
