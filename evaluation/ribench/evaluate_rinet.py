"""Evaluate RINet (CNN) on the full RIbench dataset."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..', '..')
sys.path.insert(0, os.path.join(_script_dir, '..'))
sys.path.insert(0, _project_root)

from evaluation.utils import compute_errors, load_full_ribench, standardize_targets, load_ribench_meta, compute_zdevs, print_results
from modeling.utils import load_model, standardize_samples, predict_ris


def main(ribench_dir=None, meta_csv=None, exclude=('CRP', 'LDH')):
    if ribench_dir is None:
        ribench_dir = os.path.join(_script_dir, '..', '..', 'data', 'RIbench', 'Data')
    if meta_csv is None:
        meta_csv = os.path.join(_script_dir, '..', '..', 'data', 'RIbench', 'SpecificationTestSets.csv')
    model_path = os.path.join(_script_dir, '..', '..', 'modeling', 'model')

    # load full RIbench dataset
    test_x, test_y, test_files, test_analytes = load_full_ribench(ribench_dir, meta_csv, exclude_analytes=exclude)

    # scale data
    data_scaled, data_means, data_stds = standardize_samples(test_x)

    # load model and predict
    model, scalery = load_model(model_path)
    test_p, test_p_ris = predict_ris(model, scalery, data_scaled)

    # get target RIs in standardized space
    test_y_scaled = standardize_targets(test_y, data_means, data_stds)

    # compute normalized errors
    errors = compute_errors(test_y_scaled, test_p_ris)

    # un-standardize predicted RIs to original space for zdev
    est_lrls = np.array([ris[0] * std + mean for ris, mean, std in zip(test_p_ris, data_means, data_stds)])
    est_urls = np.array([ris[1] * std + mean for ris, mean, std in zip(test_p_ris, data_means, data_stds)])

    # compute z-score deviations
    meta = load_ribench_meta(meta_csv)
    zdevs = compute_zdevs(test_files, est_lrls, est_urls, meta)

    print_results("RINet on RIbench", errors, zdevs, test_analytes)

    return errors, zdevs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ribench_dir', type=str, default=None,
                        help='Path to RIbench Data/ directory (default: data/RIbench/Data/)')
    parser.add_argument('--meta_csv', type=str, default=None,
                        help='Path to RIbench metadata CSV (default: data/RIbench/SpecificationTestSets.csv)')
    parser.add_argument('--exclude', type=str, nargs='+', default=['CRP', 'LDH'],
                        help='Analytes to exclude (default: CRP LDH). Use --exclude none to include all.')
    args = parser.parse_args()
    exclude = [] if args.exclude == ['none'] else args.exclude
    main(ribench_dir=args.ribench_dir, meta_csv=args.meta_csv, exclude=tuple(exclude))
