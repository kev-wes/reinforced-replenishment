import numpy as np
from spec_metric import spec as spec_func

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """Simple error"""
    return actual - predicted


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """Naive forecasting method which just repeats previous samples"""
    return actual[:-seasonality]


def mae(actual: np.ndarray, predicted: np.ndarray):
    """Mean Absolute Error"""
    return np.mean(np.abs(_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return np.mean(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def smape_normalized(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return np.mean(
        np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )


def wape(actual: np.ndarray, predicted: np.ndarray):
    """Weighted average percentage error"""
    return np.absolute(actual - predicted).sum() / np.absolute(actual).sum()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Square Error"""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def rmsse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Scaled Error (RMSSE).
    """
    return np.sqrt(np.mean((predicted - actual) ** 2) / np.mean((actual[1:] - actual[:-1]) ** 2))


def spec(actual: np.ndarray, predicted: np.ndarray, a1=0.75, a2=0.25) -> float:
    """
    Calculates the error metric SPEC on given data

    a1: weight of the storage costs
    a2: weight of the opportunity costs

    :param actual:
    :param predicted:
    :param a1:
    :param a2:
    :return:
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    return spec_func(actual, predicted, a1=a1, a2=a2)


def sc_spec(actual: np.ndarray, predicted: np.ndarray, a1=0.75, a2=0.25):
    """
    Calculates the error metric scaled SPEC (SC_SPEC) on given data

    a1: weight of the storage costs
    a2: weight of the opportunity costs

    :param actual:
    :param predicted:
    :param a1:
    :param a2:
    :return:
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    if isinstance(actual, float):
        actual = np.array([actual])
        predicted = np.array([predicted])

    ref_predicted = np.repeat(actual.sum() / actual.__len__(), actual.__len__())
    spec_on_ref_prediction = spec(actual, ref_predicted, a1=a1, a2=a2)
    if spec_on_ref_prediction == 0:
        spec_on_ref_prediction = actual[0]
        # if only one value is present, scale spec by actual
    spec_non_scaled = spec(actual, predicted, a1=a1, a2=a2)
    sc_spec_value = spec_non_scaled / spec_on_ref_prediction
    return sc_spec_value


def maape(predicted: np.ndarray, actual: np.ndarray):
    return np.arctan(np.abs((predicted - actual) / actual)).mean()
