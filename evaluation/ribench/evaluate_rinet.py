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

from evaluation.utils import compute_errors, load_full_ribench, subsample_ribench_indices, load_ribench_meta, compute_zdevs, print_results, update_summary_scores
from modeling.utils import load_model, standardize_samples, predict_ris, extract_features


def main(ribench_dir, meta_csv, exclude=('CRP', 'LDH'), filter_positive=False, n_samples=None, seed=42):
    model_path = os.path.join(_script_dir, '..', '..', 'modeling', 'model')

    # determine which file indices to load (subsample if requested)
    file_indices = subsample_ribench_indices(meta_csv, exclude, n_samples, seed)

    # load RIbench dataset (only selected indices)
    test_x, test_y, test_files, test_analytes = load_full_ribench(
        ribench_dir, meta_csv, exclude_analytes=exclude, file_indices=file_indices)

    # load model
    print("Loading model...")
    model, scalery = load_model(model_path)

    # predict per-sample with progress
    n = len(test_x)
    est_lrls = np.zeros(n)
    est_urls = np.zeros(n)
    data_means = np.zeros(n)
    data_stds = np.zeros(n)
    print(f"Running inference on {n} samples...")
    for i, x in enumerate(test_x):
        x = np.array(x)
        if filter_positive:
            x = x[x > 0]
        mean, std = x.mean(), x.std()
        x_scaled = (x - mean) / std
        feat = extract_features(x_scaled).reshape(1, -1, 1)
        pred = model.predict(feat, verbose=0)
        pred = scalery.inverse_transform(pred)[0]
        est_lrls[i] = pred[1] * std + mean
        # CRP: use 95th percentile (index 12); others: 97.5th (index -3)
        if test_analytes[i] == 'CRP':
            est_urls[i] = pred[12] * std + mean
        else:
            est_urls[i] = pred[-3] * std + mean
        data_means[i] = mean
        data_stds[i] = std
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}")

    # For single-limit analytes (e.g. CRP), fix LRL=0 in GT and predictions
    single_limit_mask = np.isnan(test_y[:, 0])
    gt_lrls = test_y[:, 0].copy()
    gt_lrls[single_limit_mask] = 0.0
    est_lrls[single_limit_mask] = 0.0

    print("Computing errors...")
    # compute errors in original space
    gt_pairs = [np.array([l, u]) for l, u in zip(gt_lrls, test_y[:, 1])]
    pred_pairs = [np.array([l, u]) for l, u in zip(est_lrls, est_urls)]
    errors = compute_errors(gt_pairs, pred_pairs)

    # compute z-score deviations
    meta = load_ribench_meta(meta_csv)
    zdevs = compute_zdevs(test_files, est_lrls, est_urls, meta)

    print_results("RINet on RIbench", errors, zdevs, test_analytes)
    update_summary_scores("RINet", "RIBench", errors, zdevs)

    return errors, zdevs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ribench_dir', type=str, required=True,
                        help='Path to RIbench Data/ directory')
    parser.add_argument('--meta_csv', type=str, required=True,
                        help='Path to RIbench metadata CSV (SpecificationTestSets.csv)')
    parser.add_argument('--exclude', type=str, nargs='+', default=['CRP', 'LDH'],
                        help='Analytes to exclude (default: CRP LDH). Use --exclude none to include all.')
    parser.add_argument('--filter-positive', action='store_true',
                        help='Filter out non-positive values before standardizing')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Subsample to N samples (default: use all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subsampling (default: 42)')
    args = parser.parse_args()
    exclude = [] if args.exclude == ['none'] else args.exclude
    main(ribench_dir=args.ribench_dir, meta_csv=args.meta_csv, exclude=tuple(exclude),
         filter_positive=args.filter_positive, n_samples=args.n_samples, seed=args.seed)
