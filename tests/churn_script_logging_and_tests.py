import os
import logging
import churn_library as cl
from pandas import errors as pde
import pandas as pd


# import churn_library_solution as cls


def test_import():
    '''
    tests data import - this example is completed for you to assist with the other tests functions
    '''
    try:
        df = cl.import_data("../data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except pde.EmptyDataError as err:
        logging.error("Testing import_data: Your data file is empty")
        raise err
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows or columns")
        raise err


def test_create_churn_col():
    """
    tests create_churn_col
    """
    df = pd.DataFrame(columns=['Attrition_Flag'])
    df.loc[0] = ["Existing Customer"]
    df.loc[1] = ["Attrited Customer"]
    df = cl.create_churn_col(df)
    try:
        assert df.Churn.loc[0] == 0
        assert df.Churn.loc[1] == 1
        logging.info(
            "Testing create_churn_col: The churn column was created successfully: SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing create_churn_col: The transformation from Attrition_Flag to Churn bool was not successful"
        )
        raise err


def test_eda():
    '''
    tests perform eda function. Mock a dataframe and check if plot files are produced.
    '''
    df = pd.DataFrame(columns=[['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
                                'Dependent_count', 'Education_Level', 'Marital_Status',
                                'Income_Category', 'Card_Category', 'Months_on_book',
                                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                                'Churn']])



def test_encoder_helper(encoder_helper):
    '''
    tests encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    tests perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    tests train_models
    '''


if __name__ == "__main__":
    pass
