import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import boxcox

CUTOFF_Z = 5.0


def norm_err(y_test, p_test):
    """
    Calculates the normalized error between a true and predicted RI.

    Arguments:
        y_test:   true RI (array of 2 values, LRL may be NaN for CRP)
        p_test:   predicted RI (array of 2 values)

    Returns:
        normalized error
    """
    y = np.array(y_test)
    p = np.array(p_test)
    valid = ~np.isnan(y) & ~np.isnan(p)
    if not np.any(valid):
        return np.nan
    y_valid = y[valid]
    p_valid = p[valid]
    # Single-limit case (e.g. CRP with only URL): treat RI as [0, URL]
    if len(y_valid) == 1:
        y_full = np.array([0.0, y_valid[0]])
        p_full = np.array([0.0, p_valid[0]])
    else:
        y_full = y_valid
        p_full = p_valid
    targets = np.array([-1, 1]) if len(y_full) == 2 else np.array([-1, 1])[valid]
    mean = y_full.mean()
    std = y_full.std()
    normalized = (p_full - mean) / std if std > 0 else p_full - mean
    return np.mean(np.abs(targets - normalized)) / 2


def compute_errors(test_y, test_p):
    """Compute normalized errors between lists of true and predicted RIs."""
    errors = np.array([norm_err(i, j) for i, j in zip(test_y, test_p)])
    return errors


def accuracy_at_threshold(errors, threshold=0.1):
    """Fraction of non-NaN errors below threshold."""
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return np.nan
    return np.sum(valid <= threshold) / len(valid)


def summarize_errors(errors, thresholds=(0.1, 0.2)):
    """Print summary statistics for errors."""
    print(f"Mean error:   {np.nanmean(errors):.4f}")
    print(f"Median error: {np.nanmedian(errors):.4f}")
    for t in thresholds:
        print(f"Accuracy (threshold={t}): {accuracy_at_threshold(errors, t):.3f}")


def per_analyte_summary(errors, test_analytes, analyte_order=None, threshold=0.1):
    """Create per-analyte summary DataFrame."""
    if analyte_order is None:
        analyte_order = sorted(set(test_analytes))
    rows = []
    for analyte in analyte_order:
        errs = errors[test_analytes == analyte]
        valid = errs[~np.isnan(errs)]
        if len(valid) == 0:
            rows.append({'Mean Err': np.nan, 'Median Err': np.nan, 'Accuracy': np.nan})
        else:
            rows.append({
                'Mean Err': np.mean(valid),
                'Median Err': np.median(valid),
                'Accuracy': np.sum(valid <= threshold) / len(valid),
            })
    return pd.DataFrame(rows, index=analyte_order)


def print_results(title, errors, zdevs, test_analytes):
    """Print standard evaluation results."""
    print(f"=== {title} ===")
    summarize_errors(errors)
    print(f"Mean |z-dev| (excl. implausible): {np.nanmean(zdevs):.4f}")
    print()
    print(per_analyte_summary(errors, test_analytes))


def load_pickle(path):
    """Load a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_ribench_test_set(data_path):
    """Load RIbench test set (x, y, files) from pickle directory."""
    test_x = load_pickle(os.path.join(data_path, 'x.pkl'))
    test_y = load_pickle(os.path.join(data_path, 'y.pkl'))
    test_files = load_pickle(os.path.join(data_path, 'files.pkl'))
    test_analytes = np.array([i.split('/')[-2] for i in test_files])
    return test_x, test_y, test_files, test_analytes


def load_full_ribench(ribench_data_dir, meta_csv, exclude_analytes=('CRP', 'LDH')):
    """Load the full RIbench dataset directly from CSVs.

    Args:
        ribench_data_dir: path to data/RIbench/Data/
        meta_csv: path to SpecificationTestSets.csv
        exclude_analytes: analytes to skip

    Returns:
        (test_x, test_y, test_files, test_analytes)
        test_files are basenames like '1_Hb_seed_123.csv'
    """
    meta = pd.read_csv(meta_csv)
    meta = meta.set_index('Index')

    files = []
    for dirpath, _, filenames in os.walk(ribench_data_dir):
        analyte = os.path.basename(dirpath)
        if analyte in exclude_analytes:
            continue
        for fname in sorted(filenames):
            if fname.endswith('.csv') and fname[0].isdigit():
                files.append(os.path.join(dirpath, fname))

    test_x = []
    test_y = []
    test_files = []
    test_analytes = []
    for i, filepath in enumerate(files):
        if i % 500 == 0:
            print(f"Loading RIbench: {i}/{len(files)}")
        fname = os.path.basename(filepath)
        file_index = int(fname.split('_')[0])
        analyte = fname.split('_')[1]
        sample = pd.read_csv(filepath, header=None).values.squeeze()
        row = meta.loc[file_index]
        test_x.append(sample)
        test_y.append(np.array([row['GT_LRL'], row['GT_URL']]))
        test_files.append(fname)
        test_analytes.append(analyte)

    print(f"Loaded {len(test_x)} RIbench samples")
    return test_x, np.array(test_y), np.array(test_files), np.array(test_analytes)


def load_simulated_test_set(data_path):
    """Load simulated test set (x, y) from pickle directory."""
    test_x = load_pickle(os.path.join(data_path, 'x_test.pkl'))
    test_y = load_pickle(os.path.join(data_path, 'y_test.pkl'))
    test_x = [np.array(i) for i in test_x]
    return test_x, test_y


def load_refiner_predictions(prediction_dir, test_files=None):
    """Load refineR prediction CSVs.

    If test_files is provided, loads predictions matching those filenames
    and returns (predictions, completed_mask).
    Otherwise loads all CSVs in the directory sorted by name.
    """
    if test_files is not None:
        test_p = []
        completed = []
        for f in test_files:
            path = os.path.join(prediction_dir, f)
            if os.path.exists(path):
                completed.append(f)
                test_p.append(pd.read_csv(path).PointEst.values)
        return np.array(test_p), completed
    else:
        files = sorted(os.listdir(prediction_dir))
        test_p = []
        for f in files:
            test_p.append(pd.read_csv(os.path.join(prediction_dir, f)).PointEst.values)
        return np.array(test_p), files


def standardize_targets(test_y, data_means, data_stds):
    """Standardize target RIs using per-sample means and stds."""
    return [(i - j) / k for i, j, k in zip(test_y, data_means, data_stds)]


def load_ribench_meta(meta_csv_path):
    """Load RIbench metadata CSV, indexed by Index column."""
    meta = pd.read_csv(meta_csv_path)
    meta = meta.set_index('Index')
    return meta


def compute_zdev(gt_lrl, gt_url, est_lrl, est_url, nonp_mu, nonp_sigma, nonp_lambda, nonp_shift, analyte):
    """Compute z-score deviations for a single test set."""

    def to_zscore(val, shift, lam, mu, sigma):
        v = val - shift
        if v <= 0:
            v = 1e-20
        return (boxcox(v, lam) - mu) / sigma

    zz_gt_lrl = to_zscore(gt_lrl, nonp_shift, nonp_lambda, nonp_mu, nonp_sigma)
    zz_gt_url = to_zscore(gt_url, nonp_shift, nonp_lambda, nonp_mu, nonp_sigma)

    if est_lrl is None or est_url is None or np.isnan(est_lrl) or np.isnan(est_url):
        return {"zz_dev_lrl": np.nan, "zz_dev_url": np.nan, "zz_dev_ov": np.nan,
                "zz_dev_abs_ov": np.nan, "zz_dev_abs_cutoff_ov": np.nan}

    zz_est_lrl = to_zscore(est_lrl, nonp_shift, nonp_lambda, nonp_mu, nonp_sigma)
    zz_est_url = to_zscore(est_url, nonp_shift, nonp_lambda, nonp_mu, nonp_sigma)

    zz_dev_lrl = zz_gt_lrl - zz_est_lrl
    zz_dev_url = zz_gt_url - zz_est_url

    if analyte == "CRP":
        zz_dev_ov = zz_dev_url
        zz_dev_abs_ov = abs(zz_dev_url)
    else:
        zz_dev_ov = (zz_dev_lrl + zz_dev_url) / 2
        zz_dev_abs_ov = (abs(zz_dev_lrl) + abs(zz_dev_url)) / 2

    zz_dev_abs_cutoff_ov = zz_dev_abs_ov if zz_dev_abs_ov <= CUTOFF_Z else np.nan

    return {
        "zz_dev_lrl": zz_dev_lrl,
        "zz_dev_url": zz_dev_url,
        "zz_dev_ov": zz_dev_ov,
        "zz_dev_abs_ov": zz_dev_abs_ov,
        "zz_dev_abs_cutoff_ov": zz_dev_abs_cutoff_ov,
    }


def compute_zdevs(test_files, est_lrls, est_urls, meta):
    """
    Compute zdev for a list of RIbench test files.

    Args:
        test_files: array of filenames like '1729_AST_seed_265.csv'
        est_lrls:   array of estimated lower RI limits (original space)
        est_urls:   array of estimated upper RI limits (original space)
        meta:       DataFrame from load_ribench_meta(), indexed by Index

    Returns:
        array of zz_dev_abs_cutoff_ov values (NaN for implausible results)
    """
    zdevs = []
    for fname, est_lrl, est_url in zip(test_files, est_lrls, est_urls):
        file_index = int(fname.split('_')[0])
        row = meta.loc[file_index]
        result = compute_zdev(
            gt_lrl=row['GT_LRL'],
            gt_url=row['GT_URL'],
            est_lrl=est_lrl,
            est_url=est_url,
            nonp_mu=row['nonp_mu'],
            nonp_sigma=row['nonp_sigma'],
            nonp_lambda=row['nonp_lambda'],
            nonp_shift=row['nonp_shift'],
            analyte=row['Analyte'],
        )
        zdevs.append(result['zz_dev_abs_cutoff_ov'])
    return np.array(zdevs)


def plot_ri_comparison(data_scaled, predicted_ris, true_ris, errors,
                       nr=4, nc=4, idx=None, figsize=(8, 8)):
    """Plot grid of histograms with predicted vs true RIs."""
    import matplotlib.pyplot as plt
    if idx is None:
        idx = np.random.choice(len(data_scaled), nr * nc)
    plt.figure(figsize=figsize)
    for c, i in enumerate(idx):
        plt.subplot(nr, nc, c + 1)
        plt.hist(data_scaled[i], np.linspace(-4, 4, 101), density=True)
        for j in predicted_ris[i]:
            plt.axvline(j, c='b')
        for j in true_ris[i]:
            plt.axvline(j, c='k', linestyle=':')
        plt.yticks([])
        plt.title(round(errors[i], 2))
    plt.tight_layout()
