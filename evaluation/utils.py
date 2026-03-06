import os
import pickle
import numpy as np
import pandas as pd


def norm_err(y_test, p_test):
    """
    Calculates the normalized error between a true and predicted RI.

    Arguments:
        y_test:   true RI (array of 2 values)
        p_test:   predicted RI (array of 2 values)

    Returns:
        normalized error
    """
    return np.mean(
        [
            np.abs(i - j) for i, j in zip(
                [-1, 1],
                (p_test - y_test.mean()) / y_test.std())
        ]
    ) / 2


def compute_errors(test_y, test_p):
    """Compute normalized errors between lists of true and predicted RIs."""
    errors = np.array([norm_err(i, j) for i, j in zip(test_y, test_p)])
    return errors


def accuracy_at_threshold(errors, threshold=0.1):
    """Fraction of errors below threshold."""
    return len(np.where(errors <= threshold)[0]) / len(errors)


def summarize_errors(errors, thresholds=(0.1, 0.2)):
    """Print summary statistics for errors."""
    print(f"Mean error:   {np.mean(errors):.4f}")
    print(f"Median error: {np.median(errors):.4f}")
    for t in thresholds:
        print(f"Accuracy (threshold={t}): {accuracy_at_threshold(errors, t):.3f}")


def per_analyte_summary(errors, test_analytes, analyte_order=None, threshold=0.1):
    """Create per-analyte summary DataFrame."""
    if analyte_order is None:
        analyte_order = sorted(set(test_analytes))
    df = pd.DataFrame({
        'Mean Err': [np.mean(errors[np.where(test_analytes == i)[0]]) for i in analyte_order],
        'Median Err': [np.median(errors[np.where(test_analytes == i)[0]]) for i in analyte_order],
        'Accuracy': [
            len(np.where(errors[np.where(test_analytes == i)[0]] <= threshold)[0]) / len(np.where(test_analytes == i)[0])
            for i in analyte_order
        ]
    }, index=analyte_order)
    return df


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
