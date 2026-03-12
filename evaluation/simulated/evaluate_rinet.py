"""Evaluate RINet (CNN) on simulated test data."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..', '..')
sys.path.insert(0, _project_root)

from evaluation.utils import compute_errors, summarize_errors, load_simulated_test_set, standardize_targets
from modeling.utils import load_model, standardize_samples, predict_ris


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'simulated')
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'modeling', 'model')

    # load data
    test_x, test_y = load_simulated_test_set(data_path)

    # scale data
    data_scaled, data_means, data_stds = standardize_samples(test_x, filter_positive=False)

    # load model and predict
    model, scalery = load_model(model_path)
    test_p, test_p_ris = predict_ris(model, scalery, data_scaled)

    # standardize target RIs and extract 2.5/97.5 percentiles
    test_y_scaled = standardize_targets(test_y, data_means, data_stds)
    test_y_ris = [i[[1, -2]] for i in test_y_scaled]

    # compute errors
    errors = compute_errors(test_y_ris, test_p_ris)

    print("=== RINet on Simulated ===")
    summarize_errors(errors)

    return errors


if __name__ == '__main__':
    main()
