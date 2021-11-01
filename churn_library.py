"""
BETTER DESCRIPTION LATER ON: Library to predict churn. First assignment
Udacity ML DevOps engineer course.
Improve the doc string after looking at some examples and best practices.
"""


# import libraries
import os
import pathlib
import typing
from os import path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import constants
import shap
import joblib
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth: typing.Union[str, pathlib.Path]) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    dataframe = pd.read_csv(pth)
    return dataframe


def create_churn_col(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    create a "churn" column from the 'Attrition_Flag" column
    input:
            dataframe: pandas dataframe

    output:
            dataframe: pandas dataframe
    """
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def perform_eda(dataframe: pd.DataFrame, out_path: str):
    """
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    """
    if not path.exists(path.join(out_path, 'images', 'eda')):
        os.makedirs(path.join(out_path, 'images', 'eda'))
    for column in constants.HISTOGRAM_COLS:
        plt.figure(figsize=(20, 10))
        axis = dataframe[column].hist()
        axis.set_title(column)
        fig = axis.get_figure()
        fig.savefig(path.join(out_path, 'images', 'eda', column + '.pdf'))

    for column in constants.VALUE_COUNT_COL:
        plt.figure(figsize=(20, 10))
        axis = dataframe[column].value_counts('normalize').plot(kind='bar')
        axis.set_title(column)
        fig = axis.get_figure()
        fig.savefig(path.join(out_path, 'images', 'eda', column + '.pdf'))

    for column in constants.DISTRIBUTION_COL:
        plt.figure(figsize=(20, 10))
        axis = sns.histplot(
            dataframe[column],
            kde=True,
            stat='density',
            linewidth=0)
        fig = axis.get_figure()
        fig.savefig(path.join(out_path, 'images', 'eda', column + '.pdf'))

    plt.figure(figsize=(20, 10))
    axis = sns.heatmap(
        dataframe.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    fig = axis.get_figure()
    fig.savefig(path.join(out_path, 'images', 'eda', 'Correlations' + '.pdf'))


def encoder_helper(
        dataframe: pd.DataFrame,
        category_lst: typing.List,
        response: str = constants.CHURN) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for category in category_lst:
        temp_category_lst = []
        temp_group = dataframe.groupby(category).mean()['Churn']
        for val in dataframe[category]:
            temp_category_lst.append(temp_group.loc[val])
        dataframe[category + '_' + response] = temp_category_lst
    return dataframe


def perform_feature_engineering(
        dataframe: pd.DataFrame,
        response: str = constants.CHURN):
    """
    input: df: pandas dataframe response: string of response name [optional argument
                                            that could be used for naming variables or
                                            index y column]

    output:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    """
    x_features = dataframe[constants.KEEP_COLS]
    y_target = dataframe[response]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(data_array, output_pth):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  tests response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: tests predictions from logistic regression
            y_test_preds_rf: tests predictions from random forest

    output:
             None
    """
    y_train = data_array[0]
    y_test = data_array[1]
    y_train_preds_lr = data_array[2]
    y_train_preds_rf = data_array[3]
    y_test_preds_lr = data_array[4]
    y_test_preds_rf = data_array[5]

    classification_reports_parameters = {
        "Random_Forest": {
            "Random Forest Train": [
                y_train,
                y_train_preds_rf],
            "Random Forest Test": [
                y_test,
                y_test_preds_rf]},
        "Logistic_Regression": {
            "Logistic Regression Train": [
                y_train,
                y_train_preds_lr],
            "Logistic Regression Test": [
                y_test,
                y_test_preds_lr]}
    }

    if not path.exists(path.join(output_pth, 'images', 'results')):
        os.makedirs(path.join(output_pth, 'images', 'results'))

    for modelling_method, parameters_dict in classification_reports_parameters.items():
        for title, data_sets in parameters_dict.items():
            plt.rc('figure', figsize=(5, 5))
            plt.text(
                0.01, 1.25, title, {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.05, str(
                    classification_report(
                        data_sets[0], data_sets[1])), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.savefig(
                path.join(
                    output_pth,
                    'images',
                    'results',
                    modelling_method +
                    '.jpg'))
            plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of x values
            output_pth: path to store the figure

    output:
             None
    """
    if not path.exists(path.join(output_pth, 'images', 'results')):
        os.makedirs(path.join(output_pth, 'images', 'results'))

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(
        path.join(
            output_pth,
            'images',
            'results',
            'Feature_Importance' +
            '.jpg'))
    plt.close()


def prediction_explainer_plot(model, x_test, output_pth):
    """
    creates and stores the prediction explainer figure in output_pth
    input:
            model: model object used for predictions
            x_test: pandas dataframe of x values
            output_pth: path to store the figure

    output:
             None
    """
    if not path.exists(path.join(output_pth, 'images', 'results')):
        os.makedirs(path.join(output_pth, 'images', 'results'))

    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.savefig(
        path.join(
            output_pth,
            'images',
            'results',
            'Prediction_Explainer' +
            '.jpg'))
    plt.close()


def train_models(x_train, x_test, y_train, y_test, output_pth):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    if not path.exists(path.join(output_pth, 'images', 'models')):
        os.makedirs(path.join(output_pth, 'images', 'models'))

    if not path.exists(path.join(output_pth, 'models')):
        os.makedirs(path.join(output_pth, 'models'))

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=10000)

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=constants.PARAM_GRID, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    data_array = [
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    ]
    classification_report_image(data_array, output_pth)

    feature_importance_plot(cv_rfc, x_test, output_pth)

    prediction_explainer_plot(cv_rfc, x_test, output_pth)

    joblib.dump(
        cv_rfc.best_estimator_,
        path.join(
            output_pth,
            'models',
            'rfc_model' +
            '.pkl'))
    joblib.dump(
        lrc,
        path.join(
            output_pth,
            'models',
            'logistic_model' +
            '.pkl'))


if __name__ == "__main__":
    data_frame = import_data("./data/bank_data.csv")
    data_frame_w_churn = create_churn_col(data_frame)
    perform_eda(data_frame_w_churn, './artifacts')
    encoded_data_frame = encoder_helper(
        data_frame_w_churn,
        constants.CATEGORY_LST,
        constants.CHURN)
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        encoded_data_frame, constants.CHURN)
    train_models(x_train_, x_test_, y_train_, y_test_, './artifacts')
