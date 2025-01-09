from enum import Enum


class ClassifierClasses(str, Enum):
    """Mapping of class labels to their string representations. (also file path location)"""

    benign = "benign"
    malignant = "malignant"
