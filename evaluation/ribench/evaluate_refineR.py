"""Evaluate refineR on RIbench test subset."""
import os
import sys
import pickle
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..', '..')
sys.path.insert(0, _project_root)

from evaluation.utils import norm_err, compute_errors, summarize_errors, per_analyte_summary


def main():
    test_data_path = os.path.join(os.path.dirname(__file__), 'ribench_sample')
    prediction_path = os.path.join(os.path.dirname(__file__), 'refineR_predictions')

    # load test data
    test_x = pickle.load(open(os.path.join(test_data_path, 'x.pkl'), 'rb'))
    test_y = np.array(pickle.load(open(os.path.join(test_data_path, 'y.pkl'), 'rb')))
    test_files = pickle.load(open(os.path.join(test_data_path, 'files.pkl'), 'rb'))
    test_files = np.array([i.split('/')[-1] for i in test_files])
    test_analytes = np.array([i.split('_')[1] for i in test_files])

    # load predictions (only for completed files)
    test_p = []
    completed = []
    for f in test_files:
        path = os.path.join(prediction_path, f)
        if os.path.exists(path):
            completed.append(f)
            test_p.append(pd.read_csv(path).PointEst.values)

    test_p = np.array(test_p)

    # filter to only completed predictions
    idx_complete = np.array([i in completed for i in test_files])
    test_x = [test_x[i] for i in np.where(idx_complete)[0]]
    test_y = test_y[idx_complete]
    test_files = test_files[idx_complete]
    test_analytes = test_analytes[idx_complete]

    # compute errors
    errors = compute_errors(test_y, test_p)

    print("=== refineR on RIbench ===")
    summarize_errors(errors)
    print()
    print(per_analyte_summary(errors, test_analytes,
                              analyte_order=['Hb', 'Ca', 'FT4', 'AST', 'LACT', 'GGT', 'TSH', 'IgE']))


if __name__ == '__main__':
    main()
