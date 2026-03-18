"""Evaluate refineR on the full RIbench dataset.

Reads ground truths from SpecificationTestSets.csv and predictions from
refineR_predictions/. Does NOT load the raw data CSVs.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..', '..')
sys.path.insert(0, _project_root)

from evaluation.utils import compute_errors, compute_zdev, print_results, subsample_ribench_indices, update_summary_scores


def main(meta_csv, exclude=(), n_samples=None, seed=42):
    prediction_path = os.path.join(_script_dir, 'refineR_predictions')

    # Determine which file indices to use (subsample if requested)
    file_indices = subsample_ribench_indices(meta_csv, exclude, n_samples, seed)

    # Load spec CSV (contains ground truths and z-score parameters)
    meta = pd.read_csv(meta_csv)
    meta = meta[meta['Index'].isin(file_indices)]

    # Match predictions to spec rows (include all rows, NaN when missing)
    test_y = []
    test_p = []
    test_analytes = []
    test_files = []
    n_missing = 0

    for _, row in meta.iterrows():
        idx = int(row['Index'])
        analyte = row['Analyte']
        start_seed = int(row['startSeed'])
        fname = f"{idx}_{analyte}_seed_{start_seed}.csv"
        pred_file = os.path.join(prediction_path, fname)

        if os.path.exists(pred_file):
            pred = pd.read_csv(pred_file)
            est_lrl = pred.PointEst.values[0]
            est_url = pred.PointEst.values[1]
        else:
            n_missing += 1
            est_lrl, est_url = np.nan, np.nan

        test_y.append(np.array([row['GT_LRL'], row['GT_URL']]))
        test_p.append(np.array([est_lrl, est_url]))
        test_analytes.append(analyte)
        test_files.append(fname)

    test_y = np.array(test_y)
    test_p = np.array(test_p)
    test_analytes = np.array(test_analytes)

    total = len(meta)
    found = len(test_files)
    print(f"Predictions found for {found}/{total} files ({n_missing} missing)")

    # Compute normalized errors
    errors = compute_errors(test_y, test_p)

    # Compute z-score deviations
    meta_indexed = meta.set_index('Index')
    zdevs = []
    for fname, est_lrl, est_url in zip(test_files, test_p[:, 0], test_p[:, 1]):
        file_index = int(fname.split('_')[0])
        row = meta_indexed.loc[file_index]
        result = compute_zdev(
            gt_lrl=row['GT_LRL'], gt_url=row['GT_URL'],
            est_lrl=est_lrl, est_url=est_url,
            nonp_mu=row['nonp_mu'], nonp_sigma=row['nonp_sigma'],
            nonp_lambda=row['nonp_lambda'], nonp_shift=row['nonp_shift'],
            analyte=row['Analyte'],
        )
        zdevs.append(result['zz_dev_abs_cutoff_ov'])
    zdevs = np.array(zdevs)

    print_results("refineR on RIbench", errors, zdevs, test_analytes)
    update_summary_scores("refineR", "RIBench", errors, zdevs)

    return errors, zdevs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True,
                        help='Path to SpecificationTestSets.csv')
    parser.add_argument('--exclude', type=str, nargs='+', default=['CRP', 'LDH'],
                        help='Analytes to exclude (default: CRP LDH). Use --exclude none to include all.')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Subsample to N rows (default: use all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subsampling (default: 42)')
    args = parser.parse_args()
    exclude = [] if args.exclude == ['none'] else args.exclude
    main(meta_csv=args.meta_csv, exclude=tuple(exclude),
         n_samples=args.n_samples, seed=args.seed)
