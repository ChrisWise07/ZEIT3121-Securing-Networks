import pandas as pd
import numpy as np
from typing import Any, Callable, List, Dict, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

COL_CHAR_LIMIT = 79

pd.set_option("display.max_colwidth", None)


def return_raw_data_frame(file_path: str) -> pd.DataFrame:
    """
    Open csv file and return a raw pd data frame

    Args:
        file_path (str): The path to the file

    Returns:
        pd.DataFrame: The raw data frame
    """
    return pd.read_csv(file_path)


def remove_reduntant_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant columns by checking if the column contains the same
    value for all rows

    Args:
        data_frame (pd.DataFrame): The data frame

    Returns:
        pd.DataFrame: The data frame with redundant columns removed
    """
    [
        data_frame.drop(column_name, axis=1, inplace=True)
        for column_name in data_frame.columns
        if data_frame[column_name].nunique() == 1
    ]
    return data_frame


def clean_data_frame(
    data_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean the data frame by removing null values and redundant columns

    Args:
        data_frame (pd.DataFrame): The data frame

    Returns:
        pd.DataFrame: The cleaned data frame
    """
    return remove_reduntant_columns(data_frame.fillna(0))


def return_sampled_data_frame(
    data_frame: pd.DataFrame,
    labels_with_sample_size: Dict[str, int],
) -> pd.DataFrame:
    """
    Return a cleaned sample data frame with null values removed and a
    sample of data for each label

    Args:
        data_frame (pd.DataFrame): The raw data frame
        label_column_names (Dict[str:int]):
            labels and corresponding sample sizes

    Returns:
        pd.DataFrame: The cleaned sampled data frame
    """
    return pd.concat(
        [
            data_frame[data_frame["label"] == label].sample(sample_size)
            for label, sample_size in labels_with_sample_size.items()
        ],
        ignore_index=True,
        sort=False,
    )


def return_one_hot_encoded_data_frame(
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
    return pd.get_dummies(
        return_label_encoded_data_frame(
            data_frame=data_frame, categorical_column_names=["label"]
        )
    )


def return_label_encoded_data_frame(
    data_frame: pd.DataFrame, categorical_column_names: List[str]
) -> pd.DataFrame:
    """
    Replacing categorical values with label encoded values

    Args:
        data_frame (pd.DataFrame): The data frame
        categorical_column_names (List[str]):
            The categorical column names

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
    data_frame: pd.DataFrame, label_column_name: str
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
        data_frame.drop(label_column_name, axis=1),
        data_frame[label_column_name],
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
    model_class: Union[DecisionTreeClassifier, SVC, RandomForestClassifier],
) -> Union[DecisionTreeClassifier, SVC, RandomForestClassifier]:
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
    model.fit(x_train, y_train.values.ravel())
    return model


def print_model_performance(
    y_pred: pd.DataFrame,
    y_test: pd.DataFrame,
    model_class: Union[DecisionTreeClassifier, SVC, RandomForestClassifier],
    encoding_type: str = None,
) -> None:
    """
    Print performance metrics using predictions and test sets

    Args:
        y_pred (pd.DataFrame): The y predictions set
        y_test (pd.DataFrame): The y test set
        model_class (class): The model
        encoding_type (str, optional):
            The encoding type. Defaults to None.
    """
    model_name_encode_string = (
        (f"{model_class.__name__} ({encoding_type} encoding)")
        if encoding_type
        else model_class.__name__
    )

    len_model_name_encode_string = len(model_name_encode_string)
    num_of_dashes = (COL_CHAR_LIMIT - len_model_name_encode_string) // 2

    print(
        f"{'-' * num_of_dashes} {model_name_encode_string} ",
        f"{'-' * num_of_dashes + '-' * ( len_model_name_encode_string % 2)}",
        f"\nAccuracy: \n{accuracy_score(y_test, y_pred)}",
        f"\nConfusion Matrix: \n{confusion_matrix(y_test, y_pred)}",
        f"\nClassification Report: \n ",
        f"{classification_report(y_test, y_pred, zero_division=0)}",
        f"\n{'-' * (num_of_dashes*2 + 2 + len_model_name_encode_string)}\n\n",
        sep="",
    )


def train_test_models_with_encodings(
    data: pd.DataFrame,
    algorithms: List[
        Union[DecisionTreeClassifier, SVC, RandomForestClassifier]
    ],
    encoding_functions_with_args: Dict[
        Callable[[pd.DataFrame], pd.DataFrame], List[Any]
    ],
    algorithm_encoding_performance_data: Dict[str, Dict[str, list]],
) -> None:
    """
    Train and test models with different encodings and update
    performance data

    Args:
        data (pd.DataFrame): The data frame
        algorithms (List[Union[DecisionTreeClassifier, SVC]]):
            The list of algorithms
        encoding_functions_with_args (
            Dict[Callable[[pd.DataFrame],
            pd.DataFrame],
            List[Any]]
        ):
            The list of encoding functions with their arguments
        algorithm_encoding_performance_data (
            Dict[str, Dict[str, float]]
        ):
            The performance data that is updated
    """
    encoding_with_train_test_split_data = {}

    for encoding_function, args in encoding_functions_with_args.items():
        encoding_with_train_test_split_data.update(
            {
                encoding_function.__name__: return_train_test_split(
                    *return_x_y_split(
                        encoding_function(data, *args),
                        ["label"],
                    ),
                )
            }
        )

    for algorithm in algorithms:
        for (
            encoding_function_name,
            train_test_split_data,
        ) in encoding_with_train_test_split_data.items():
            x_train, x_test, y_train, y_test = train_test_split_data

            y_pred = return_train_model(x_train, y_train, algorithm).predict(
                x_test
            )

            report = classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            )

            algorithm_encoding_performance_data[
                f"{algorithm.__name__} ({encoding_function_name})"
            ]["accuracy"].append(report["accuracy"])

            algorithm_encoding_performance_data[
                f"{algorithm.__name__} ({encoding_function_name})"
            ]["f1-scores"].append(report["weighted avg"]["f1-score"])


def train_and_return_performance_data_for_given_ratio(
    number_of_normal: int,
    number_of_attack: int,
    data: pd.DataFrame,
    num_of_epochs: int,
) -> Tuple[float, float, float]:
    """
    Train and return performance data for given ratio

    Args:
        number_of_normal (int): The number of normal data points
        number_of_attack (int): The number of attack data points
        data (pd.DataFrame): The data frame
        num_of_epochs (int): The number of epochs

    Returns:
        tuple:
            The performance data in form of:
                [normal_f1_score, attack_f1_score, accuracy]
    """
    normal_f1_scores = []
    attack_f1_scores = []
    accuracy_scores = []

    for _ in range(num_of_epochs + 1):
        sampled_data = return_sampled_data_frame(
            data_frame=data,
            labels_with_sample_size={0: number_of_normal, 1: number_of_attack},
        )

        (x_train, x_test, y_train, y_test) = return_train_test_split(
            *return_x_y_split(
                sampled_data,
                "label",
            ),
        )

        y_pred = return_train_model(
            x_train, y_train, DecisionTreeClassifier
        ).predict(x_test)

        report = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )

        normal_f1_scores.append(report["0"]["f1-score"])
        attack_f1_scores.append(report["1"]["f1-score"])
        accuracy_scores.append(report["accuracy"])

    return (
        np.average(normal_f1_scores),
        np.average(attack_f1_scores),
        np.average(accuracy_scores),
    )


def return_data_on_classification_ratio(
    data: pd.DataFrame,
    number_of_data_points: int = 19,
    total_number_of_records: int = 2000,
    num_of_epochs: int = 20,
) -> pd.DataFrame:
    """
    Return the data on classification ratio

    Args:
        data (pd.DataFrame): The data frame
        number_of_data_points (int, optional):
            The number of data points. Defaults to 19.
        total_number_of_records (int, optional):
            The total number of records. Defaults to 2000.
        num_of_epochs (int, optional):
            The number of epochs. Defaults to 20.

    Returns:
        pd.DataFrame: The data on classification ratio
    """
    list_of_normal_data_representation_ratios = np.linspace(
        0.0, 1.0, number_of_data_points + 1, endpoint=False
    )[1:]
    normal_f1_scores = []
    attack_f1_scores = []
    accuracy_scores = []

    for ratio in list_of_normal_data_representation_ratios:
        (
            normal_f1_score,
            attack_f1_score,
            accuracy_score,
        ) = train_and_return_performance_data_for_given_ratio(
            number_of_normal=int(ratio * total_number_of_records),
            number_of_attack=int((1 - ratio) * total_number_of_records),
            data=data,
            num_of_epochs=num_of_epochs,
        )

        normal_f1_scores.append(normal_f1_score)
        attack_f1_scores.append(attack_f1_score)
        accuracy_scores.append(accuracy_score)

    return pd.DataFrame(
        {
            "Ratio of Normal Data": list_of_normal_data_representation_ratios,
            "Normal F1 Score": normal_f1_scores,
            "Attack F1 Score": attack_f1_scores,
            "Accuracy Score": accuracy_scores,
        }
    )


def bench_mark_encoding_and_algorithms_wrapper(
    file_path: str, epochs: int
) -> None:
    """
    Return performance data (average accuracy and f1 score) for
    different encoding methods and algorithms

    Args:
        file_path (str): The file path
        epochs (int):
            The number of repetitions of training and testing
    """
    algorithm_encoding_performance_data = {}

    algorithms = [
        DecisionTreeClassifier,
        SVC,
        RandomForestClassifier,
    ]

    encoding_functions_with_args = {
        return_one_hot_encoded_data_frame: [],
        return_label_encoded_data_frame: [["flgs", "dir", "state", "label"]],
    }

    [
        algorithm_encoding_performance_data.update(
            {
                f"{algorithm.__name__} ({encoding_function.__name__})": {
                    "accuracy": [],
                    "f1-scores": [],
                },
            }
        )
        for algorithm in algorithms
        for encoding_function in encoding_functions_with_args.keys()
    ]

    for _ in range(epochs):
        data = return_sampled_data_frame(
            clean_data_frame(data_frame=return_raw_data_frame(file_path)),
            {"attack": 2000, "normal": 2000},
        )
        train_test_models_with_encodings(
            data,
            algorithms,
            encoding_functions_with_args,
            algorithm_encoding_performance_data,
        )

    average_algorithm_encoding_performance_data = {}

    [
        average_algorithm_encoding_performance_data.update(
            {
                f"{key}": {
                    "average_accuracy": np.average(value["accuracy"]),
                    "average_f1_score": np.average(value["f1-scores"]),
                }
            }
        )
        for key, value in algorithm_encoding_performance_data.items()
    ]

    print(
        pd.DataFrame(average_algorithm_encoding_performance_data).transpose()
    )


def return_data_on_classification_ratio_wrapper(
    file_path: str, num_of_epochs: int
) -> None:
    """
    Wrapper function for return_data_on_classification_ratio. Will
    print the data on classification ratio as pandas dataframe

    Args:
        file_path (str): The file path
        number_of_epochs (int): The number of epochs for training
    """
    print(
        return_data_on_classification_ratio(
            return_one_hot_encoded_data_frame(
                clean_data_frame(data_frame=return_raw_data_frame(file_path)),
            ),
            num_of_epochs=num_of_epochs,
        ).to_string(index=False)
    )


def final_performance_data_wrapper(file_path: str, epochs: int) -> None:
    """
    Wrapper function for final_performance_data. Will print the
    performance data for the best final model

    Args:
        file_path (str): The file path
        epochs (int): The number of epochs
    """
    best_model_accuracy = 0.0

    for _ in range(epochs):
        x_train, x_test, y_train, y_test = return_train_test_split(
            *return_x_y_split(
                return_one_hot_encoded_data_frame(
                    clean_data_frame(
                        return_sampled_data_frame(
                            return_raw_data_frame(file_path),
                            {"attack": 2000, "normal": 2000},
                        )
                    ),
                ),
                ["label"],
            )
        )

        model = return_train_model(x_train, y_train, RandomForestClassifier)
        model_accuracy_score = accuracy_score(y_test, model.predict(x_test))

        if model_accuracy_score > best_model_accuracy:
            best_model_accuracy = model_accuracy_score
            best_model = model
            best_x_test = x_test
            best_y_test = y_test

    print_model_performance(
        y_pred=best_model.predict(best_x_test),
        y_test=best_y_test,
        model_class=RandomForestClassifier,
    )


def main() -> None:
    """
    Main function
    """
    file_path = "network_flow_data.csv"
    epochs = 20

    print("\n---Bench Mark Encoding and Algorithms---")
    bench_mark_encoding_and_algorithms_wrapper(file_path, epochs)

    print("\n\n\n---Data on Classification Ratio---")
    return_data_on_classification_ratio_wrapper(file_path, epochs)

    print("\n\n\n---Final Performance Data---")
    final_performance_data_wrapper(file_path, epochs)


if __name__ == "__main__":
    main()
