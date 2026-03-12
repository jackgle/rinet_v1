"""Evaluate refineR on simulated test data."""
import os
import sys
import pickle
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..', '..')
sys.path.insert(0, _project_root)

from evaluation.utils import norm_err, compute_errors, summarize_errors


def main():
    test_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'simulated')
    prediction_path = os.path.join(os.path.dirname(__file__), 'refineR_predictions')

    # load test data
    test_x = pickle.load(open(os.path.join(test_data_path, 'x_test.pkl'), 'rb'))
    test_y = pickle.load(open(os.path.join(test_data_path, 'y_test.pkl'), 'rb'))

    # load predictions
    test_files = sorted(os.listdir(prediction_path))
    test_p = []
    for f in test_files:
        test_p.append(pd.read_csv(os.path.join(prediction_path, f)).PointEst.values)

    # align test data with prediction files
    test_indices = [int(i.split('.')[0]) for i in test_files]
    test_y = [test_y[i][[1, -2]] for i in test_indices]
    test_x = [np.array(test_x[i]) for i in test_indices]
    test_p = [i + min(j) for i, j in zip(test_p, test_x)]

    test_p = np.array(test_p)
    test_y = np.array(test_y)

    # compute errors, filtering NaNs
    errors = np.array([norm_err(i, j) for i, j in zip(test_y, test_p)])
    idx = ~np.isnan(errors)
    errors = errors[idx]

    print("=== refineR on Simulated ===")
    summarize_errors(errors)

    return errors


if __name__ == '__main__':
    main()
