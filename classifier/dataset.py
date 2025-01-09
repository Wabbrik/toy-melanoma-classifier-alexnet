from enum import Enum
from os import listdir
from os.path import join

import kagglehub

KAGGLE_SRC = "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"


class ClassifierClasses(str, Enum):
    """Mapping of class labels to their string representations. (also file path location)"""

    benign = "Benign"
    malignant = "Malignant"


def fetch_dataset() -> str:
    return join(kagglehub.dataset_download(KAGGLE_SRC), "melanoma_cancer_dataset")


def test_dataset() -> str:
    return join(fetch_dataset(), "test")


def train_dataset() -> str:
    return join(fetch_dataset(), "train")


def class_data(dataset: str, cls: ClassifierClasses) -> str:
    """
    Returns the path to the training data for the specified class.

    Example:
    >>> class_data(train_dataset(), ClassifierClasses.benign)
    '/path/to/train/benign'
    """
    return join(dataset, cls.name)


def labeled_dataset(dataset: str) -> dict[str, str]:
    """
    Returns a dictionary of paths to class labels for the specified dataset.
    """
    labeled_files = {}

    for cls in ClassifierClasses:
        data_path = class_data(dataset, cls)
        for path in listdir(data_path):
            labeled_files[join(data_path, path)] = cls.name

    return labeled_files
