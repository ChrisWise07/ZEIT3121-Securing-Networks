import numpy as np
import pandas as pd

from typing import List, Dict, Tuple, Set, Optional
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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


def return_cleaned_data_frame(raw_data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw pd data frame by replacing missing values with zeros and
    replacing categorical values with one-hot encoded values

    Args:
        raw_data_frame (pd.DataFrame): The raw data frame

    Returns:
        pd.DataFrame: The cleaned data frame

    """
    return pd.get_dummies(raw_data_frame.fillna(0))


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


def main():
    """
    Main function
    """
    file_path = "network_flow_data.csv"

    # Get the x and y sets
    x, y = return_x_y_split(
        data_frame=return_cleaned_data_frame(
            raw_data_frame=return_raw_data_frame(file_path)
        ),
        label_column_names=["label_attack", "label_normal"],
    )

    # Get the train and test sets
    x_train, x_test, y_train, y_test = return_train_test_split(x, y)

    model = DecisionTreeClassifier()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

    print("Classification Report:", classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
