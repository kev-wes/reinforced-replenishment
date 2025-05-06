import numpy as np
import pandas as pd
import scipy.stats


def percentage_error(x: np.ndarray, y: np.ndarray):
    error = x / y
    negative = error <= 1
    error[negative] = error[negative] - 1
    error[~negative] = 1 - error[~negative] ** -1
    return error


def t_test(
    x: np.ndarray, xmean: float, confidence_level: float | int = 0.95
) -> tuple[str, float]:
    df = x.size - 1
    sd = np.sqrt(np.var(x, ddof=1) / x.size)
    xmean_quantile = 1 - (1 - scipy.stats.t.cdf(xmean, df=df, scale=sd)) * 2  # t-test
    if xmean_quantile >= confidence_level:
        half_width = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, df=df, scale=sd)
        lower = xmean - half_width
        upper = xmean + half_width
        confidence_interval = f" ({lower:.1%} - {upper:.1%})"
    else:
        confidence_interval = ""
    pvalue = 1 - xmean_quantile
    return confidence_interval, pvalue


def compare_benchmarks(
    df: pd.DataFrame, name1: str, name2: str, confidence_level=0.95
) -> str:
    measures1 = df[name1].array
    measures2 = df[name2].array
    advantage = percentage_error(measures2, measures1)
    advantage_mean = advantage.mean()
    if advantage_mean >= 0:
        better = name1
        worse = name2
    else:
        better = name2
        worse = name1
        advantage_mean *= -1
    confidence_interval, pvalue = t_test(
        x=advantage, xmean=advantage_mean, confidence_level=confidence_level
    )
    return f"{better} is {advantage_mean:.1%}{confidence_interval} better than {worse} (p-value {pvalue:.3g})."


def diebold_mariano_test(df: pd.DataFrame, name1: str, name2: str, crit="MSE"):
    """
    Perform Diebold-Mariano test for comparing predictive accuracy of two forecasts.

    Parameters:
    df (pd.DataFrame): Dataframe with forecast and truth values.
    name1 (str): First forecast column name.
    name2 (str): Second forecast column name.
    crit (str): Criterion to use, default is 'MSE' (Mean Squared Error)

    Returns:
    DM statistic and p-value
    """
    e1 = df.truth.values - df[name1].values
    e2 = df.truth.values - df[name2].values

    if crit == "MSE":
        d = e1**2 - e2**2
    elif crit == "MAD":
        d = np.abs(e1) - np.abs(e2)
    elif crit == "MAPE":
        d = (np.abs(e1) / df.truth.values) - (np.abs(e2) / df.truth.values)
    else:
        raise ValueError("Criterion not recognized. Use 'MSE', 'MAD', or 'MAPE'.")

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    try:
        DM_stat = mean_d / np.sqrt(var_d / len(d))
    except ZeroDivisionError:
        DM_stat = np.NaN

    p_value = 2 * scipy.stats.norm.cdf(-np.abs(DM_stat))

    return DM_stat, p_value
