import numpy as np


def norm_err(y_test, p_test):
    """
    Calculates the normalized error between a true and predicted RIs

    Arguments:
        y_test:   true RI
        p_test:   predicted RI

    Returns:
        normalized error
    """
    return np.mean(
        [
            np.abs(i-j) for i, j in zip(
                [-1, 1],
                (p_test-y_test.mean())/y_test.std())
        ]
    )/2

