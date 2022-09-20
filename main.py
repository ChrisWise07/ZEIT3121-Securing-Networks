import pandas as pd
import pprint as pp

from typing import List, Dict, Tuple, Set, Optional
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def return_raw_data_frame(file_path: str) -> pd.DataFrame:
    """
    Open csv file and return a raw pd data frame

    Args:
        file_path (str): The path to the file

    Returns:
        pd.DataFrame: The raw data frame
    """
    return pd.read_csv(file_path)


def return_one_hot_encoded_cleaned_data_frame(
    raw_data_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean a raw pd data frame by replacing missing values with zeros and
    replacing categorical values with one-hot encoded values

    Args:
        raw_data_frame (pd.DataFrame): The raw data frame

    Returns:
        pd.DataFrame: The cleaned data frame

    """
    return pd.get_dummies(raw_data_frame.fillna(0))


def return_label_encoded_cleaned_data_frame(
    raw_data_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean a raw pd data frame by replacing missing values with zeros and
    replacing categorical values with one-hot encoded values

    Args:
        raw_data_frame (pd.DataFrame): The raw data frame

    Returns:
        pd.DataFrame: The cleaned data frame

    """
    return raw_data_frame.fillna(0).apply(
        lambda x: x.astype("category").cat.codes
    )


def return_x_y_split(
    data_frame: pd.DataFrame, label_column_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data frame into x and y sets

    Args:
        data_frame (pd.DataFrame): The data frame
        label_column_name (str): The name of the label column

    Returns:
        tuple: The x and y sets
    """
    return (
        data_frame.drop(label_column_names, axis=1).values,
        data_frame[label_column_names.pop()].values,
    )


def return_train_test_split(
    x: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the x and y sets into train and test sets

    Args:
        x (pd.DataFrame): The x set
        y (pd.DataFrame): The y set
        test_size (float, optional): The size of the test set. Defaults to 0.2.

    Returns:
        tuple: The train and test sets
    """
    return train_test_split(x, y, test_size=test_size)


def train_model_and_print_performance(
    data: List[pd.DataFrame],
    encoding_type: str,
    model_class: DecisionTreeClassifier,
) -> None:
    """
    Train a model and print performance metrics

    Args:
        x_train (pd.DataFrame): The x train set
        y_train (pd.DataFrame): The y train set
        x_test (pd.DataFrame): The x test set
        y_test (pd.DataFrame): The y test set
    """
    x_train, x_test, y_train, y_test = data

    model = model_class()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    model_name_encode_string = (
        f"{model_class.__name__} ({encoding_type} encoding)"
    )
    len_model_name_encode_string = len(model_name_encode_string)
    num_of_dashes = (80 - len_model_name_encode_string) // 2

    print(
        f"""{"-" * num_of_dashes} {model_name_encode_string} {"-" * num_of_dashes + "-" * ( len_model_name_encode_string % 2)}
        \nAccuracy: \n{accuracy_score(y_test, y_pred)}
        \nConfusion Matrix: \n{confusion_matrix(y_test, y_pred)} 
        \nClassification Report: \n{classification_report(y_test, y_pred, zero_division=0)}
        \n{"-" * (num_of_dashes*2 + 2 + len_model_name_encode_string)}\n\n"""
    )


def main():
    """
    Main function
    """
    file_path = "network_flow_data.csv"

    raw_data_frame = return_raw_data_frame(file_path)

    one_hot_encoded_data = return_train_test_split(
        *return_x_y_split(
            data_frame=return_one_hot_encoded_cleaned_data_frame(
                raw_data_frame
            ),
            label_column_names=["label_attack", "label_normal"],
        )
    )

    label_encoded_data = return_train_test_split(
        *return_x_y_split(
            data_frame=return_label_encoded_cleaned_data_frame(raw_data_frame),
            label_column_names=["label"],
        )
    )

    [
        train_model_and_print_performance(data, encoding_type, model_class)
        for model_class in [DecisionTreeClassifier, SVC]
        for data, encoding_type in [
            (one_hot_encoded_data, "one-hot"),
            (label_encoded_data, "label"),
        ]
    ]


if __name__ == "__main__":
    main()
