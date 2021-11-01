"""
conftest file with fixtures
"""
import typing
import logging
import pytest
import constants
import pandas as pd
import churn_library as cl
from pandas import errors as pde


@pytest.fixture(name='test_dataframe')
def test_dataframe_() -> pd.DataFrame:
    """Use fixtures to define the dataframe"""
    try:
        test_data = cl.import_data('./data/bank_data_test.csv')
        logging.info("Fixture: Test data imported successfully: SUCCESS")
    except pde.EmptyDataError as err:
        logging.error("Fixture: Testing import_data: Your data file is empty")
        raise err
    except FileNotFoundError as err:
        logging.error("Fixture: Testing import_data: The file wa    sn't found")
        raise err
    return test_data


@pytest.fixture(name='test_dataframe_w_churn')
def test_dataframe_w_churn_(test_dataframe) -> pd.DataFrame:
    """Function to add the churn column"""
    try:
        test_df = test_dataframe
        return_df = cl.create_churn_col(test_df)
        assert return_df is not None
        logging.info(
            'Fixture: Churn column added successfully: SUCCESS'
        )
    except AssertionError as err:
        logging.error(
            'Fixture: The churn column was not created'
        )
        raise err
    return return_df


@pytest.fixture(name='test_dataframe_encoded')
def test_dataframe_encoded_(test_dataframe_w_churn) -> pd.DataFrame:
    """Function to add the encode categorical columns"""

    try:
        test_dataframe_encoded = cl.encoder_helper(
            test_dataframe_w_churn, constants.CATEGORY_LST, constants.CHURN)
        logging.info(
            'Fixture: The categorical columns were encoded: SUCCESS'
        )
    except KeyError as err:
        logging.error(
            "Fixture: One of the categorical columns is not present in the test dataframe")
        raise err
    return test_dataframe_encoded


@pytest.fixture(name='features')
def test_engineered_features(test_dataframe_encoded) -> typing.Tuple:
    """Function to perform feature engineering"""

    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            test_dataframe_encoded)
        for data_set in [x_train, x_test, y_train, y_test]:
            assert data_set is not None

        logging.info(
            'Fixture: Features engineered: SUCCESS'
        )
    except AssertionError as err:
        logging.error(
            "Fixture: Creation of engineered features failed. "
        )
        raise err

    return x_train, x_test, y_train, y_test
