import pandas as pd
from typing import List, Dict, Tuple, Union
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


def return_cleaned_sampled_data_frame(
    raw_data_frame: pd.DataFrame,
    labels_with_sample_size: Dict[str, int],
) -> pd.DataFrame:
    """
    Return a cleaned sample data frame with null values removed and a
    sample of data for each label

    Args:
        raw_data_frame (pd.DataFrame): The raw data frame
        label_column_names (Dict[str:int]):
            labels and corresponding sample sizes

    Returns:
        pd.DataFrame: The cleaned sampled data frame
    """
    return pd.concat(
        [
            raw_data_frame[raw_data_frame["label"] == label].sample(
                sample_size
            )
            for label, sample_size in labels_with_sample_size.items()
        ],
        ignore_index=True,
        sort=False,
    )


def return_one_hot_encoded_cleaned_data_frame(
    data_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replacing categorical values with one-hot encoded values

    Args:
        data_frame (pd.DataFrame): The data frame

    Returns:
        pd.DataFrame:
            The data frame with one-hot encoded categorical values
    """
    return pd.get_dummies(data_frame)


def return_label_encoded_cleaned_data_frame(
    data_frame: pd.DataFrame, categorical_column_names: List[str]
) -> pd.DataFrame:
    """
    Replacing categorical values with label encoded values

    Args:
        data_frame (pd.DataFrame): The data frame

    Returns:
        pd.DataFrame:
            The data frame with label encoded categorical values

    """
    from sklearn.preprocessing import LabelEncoder

    number = LabelEncoder()

    for column_name in categorical_column_names:
        data_frame[column_name] = number.fit_transform(
            data_frame[column_name].astype("str")
        )

    return data_frame


def return_x_y_split(
    data_frame: pd.DataFrame, label_column_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data frame into x and y sets

    Args:
        data_frame (pd.DataFrame): The data frame
        label_column_name (str): The name of the label column

    Returns:
        tuple: The x and y sets in form of [x, y]
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
        test_size (float, optional):
            The size of the test set. Defaults to 0.2.

    Returns:
        tuple:
            The train and test sets in form of:
                [x_train, x_test, y_train, y_test]
    """
    return train_test_split(x, y, test_size=test_size)


def return_train_model(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_class: Union[DecisionTreeClassifier, SVC],
) -> Union[DecisionTreeClassifier, SVC]:
    """
    Train a model

    Args:
        x_train (pd.DataFrame): The x training set
        y_train (pd.DataFrame): The y training set
        model_class (class): The model class

    Returns:
        model: The trained model
    """
    model = model_class()
    model.fit(x_train, y_train)
    return model


def print_model_performance(
    y_pred: pd.DataFrame,
    y_test: pd.DataFrame,
    model_class: Union[DecisionTreeClassifier, SVC],
    encoding_type: str = None,
) -> None:
    """
    Print performance metrics using predictions and test sets

    Args:
        y_pred (pd.DataFrame): The y predictions set
        y_test (pd.DataFrame): The y test set
    """
    model_name_encode_string = (
        (f"{model_class.__name__} ({encoding_type} encoding)")
        if encoding_type
        else model_class.__name__
    )

    len_model_name_encode_string = len(model_name_encode_string)
    num_of_dashes = (80 - len_model_name_encode_string) // 2

    print(
        f"{'-' * num_of_dashes} {model_name_encode_string}",
        f"{'-' * num_of_dashes + '-' * ( len_model_name_encode_string % 2)}",
        f"\nAccuracy: \n{accuracy_score(y_test, y_pred)}",
        f"\nConfusion Matrix: \n{confusion_matrix(y_test, y_pred)}",
        f"\nClassification Report: \n ",
        f"{classification_report(y_test, y_pred, zero_division=0)}",
        f"\n{'-' * (num_of_dashes*2 + 2 + len_model_name_encode_string)}\n\n",
        sep="",
    )


def main():
    """
    Main function
    """
    file_path = "network_flow_data.csv"

    sampled_cleaned_data = return_cleaned_sampled_data_frame(
        raw_data_frame=return_raw_data_frame(file_path),
        labels_with_sample_size={"normal": 2000, "attack": 2000},
    )

    one_hot_encoded_cleaned_data_frame = (
        return_one_hot_encoded_cleaned_data_frame(
            data_frame=sampled_cleaned_data
        )
    )

    label_encoded_cleaned_data_frame = return_label_encoded_cleaned_data_frame(
        data_frame=sampled_cleaned_data,
        categorical_column_names=["label"],
    )

    (
        x_test_one_hot_encoded,
        x_train_one_hot_encoded,
        y_test_one_hot_encoded,
        y_train_one_hot_encoded,
    ) = return_train_test_split(
        *return_x_y_split(
            one_hot_encoded_cleaned_data_frame,
            ["label_normal", "label_attack"],
        ),
    )

    (
        x_test_label_encoded,
        x_train_label_encoded,
        y_test_label_encoded,
        y_train_label_encoded,
    ) = return_train_test_split(
        *return_x_y_split(
            label_encoded_cleaned_data_frame,
            ["label"],
        ),
    )

    [
        print_model_performance(
            y_pred=return_train_model(x_train, y_train, model_class).predict(
                x_test
            ),
            y_test=y_test,
            model_class=model_class,
            encoding_type=encoding_type,
        )
        for model_class in [DecisionTreeClassifier, SVC]
        for x_train, x_test, y_train, y_test, encoding_type in [
            [
                x_train_one_hot_encoded,
                x_test_one_hot_encoded,
                y_train_one_hot_encoded,
                y_test_one_hot_encoded,
                "one hot encoded",
            ],
            [
                x_train_label_encoded,
                x_test_label_encoded,
                y_train_label_encoded,
                y_test_label_encoded,
                "label encoded",
            ],
        ]
    ]


if __name__ == "__main__":
    main()
