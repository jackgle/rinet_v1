"""Evaluate RINet (CNN) on RIbench test subset."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..', '..')
sys.path.insert(0, os.path.join(_script_dir, '..'))
sys.path.insert(0, _project_root)

from evaluation.utils import compute_errors, summarize_errors, per_analyte_summary, load_ribench_test_set, standardize_targets
from modeling.utils import load_model, standardize_samples, predict_ris


def main():
    data_path = os.path.join(os.path.dirname(__file__), 'ribench_sample')
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'modeling', 'model')

    # load data
    test_x, test_y, test_files, test_analytes = load_ribench_test_set(data_path)

    # scale data
    data_scaled, data_means, data_stds = standardize_samples(test_x)

    # load model and predict
    model, scalery = load_model(model_path)
    test_p, test_p_ris = predict_ris(model, scalery, data_scaled)

    # get target RIs in standardized space
    test_y_scaled = standardize_targets(test_y, data_means, data_stds)

    # compute errors
    errors = compute_errors(test_y_scaled, test_p_ris)

    print("=== RINet on RIbench ===")
    summarize_errors(errors)
    print()
    print(per_analyte_summary(errors, test_analytes,
                              analyte_order=['Hb', 'Ca', 'FT4', 'AST', 'LACT', 'GGT', 'TSH', 'IgE']))


if __name__ == '__main__':
    main()
