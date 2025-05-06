import numpy as np


def _accuracies(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Calculates Accuracy for each sample in actual/predicted by a formula requested
    and provided by Rockwool.
    Returns values between 0 and 100.
    """
    return (1 - np.absolute(actual - predicted) / actual) * 100


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculates mean_accuracy by a formula requested and provided by Rockwool.
    Returns value between 0 and 100.
    """
    return np.mean(_accuracies(actual=actual, predicted=predicted))  # type: ignore
