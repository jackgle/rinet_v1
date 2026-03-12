"""Run all evaluations and generate summary_scores.csv."""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..')
sys.path.insert(0, _project_root)

from evaluation.utils import accuracy_at_threshold

from evaluation.ribench.evaluate_rinet import main as eval_rinet_ribench
from evaluation.simulated.evaluate_rinet import main as eval_rinet_simulated
from evaluation.ribench.evaluate_refineR import main as eval_refiner_ribench
from evaluation.simulated.evaluate_refineR import main as eval_refiner_simulated


def main(ribench_dir=None, meta_csv=None):
    results = []

    print("\n" + "=" * 50)
    errors, zdevs = eval_rinet_ribench(ribench_dir=ribench_dir, meta_csv=meta_csv)
    results.append(('Accuracy', 'RINet', 'RIBench', accuracy_at_threshold(errors, 0.1)))
    results.append(('Average Norm. Error', 'RINet', 'RIBench', np.mean(errors)))
    results.append(('Mean |Z-dev|', 'RINet', 'RIBench', np.nanmean(zdevs)))

    print("\n" + "=" * 50)
    errors = eval_rinet_simulated()
    results.append(('Accuracy', 'RINet', 'Simulated', accuracy_at_threshold(errors, 0.1)))
    results.append(('Average Norm. Error', 'RINet', 'Simulated', np.mean(errors)))

    print("\n" + "=" * 50)
    errors, zdevs = eval_refiner_ribench(ribench_dir=ribench_dir, meta_csv=meta_csv)
    results.append(('Accuracy', 'refineR', 'RIBench', accuracy_at_threshold(errors, 0.1)))
    results.append(('Average Norm. Error', 'refineR', 'RIBench', np.mean(errors)))
    results.append(('Mean |Z-dev|', 'refineR', 'RIBench', np.nanmean(zdevs)))

    print("\n" + "=" * 50)
    errors = eval_refiner_simulated()
    results.append(('Accuracy', 'refineR', 'Simulated', accuracy_at_threshold(errors, 0.1)))
    results.append(('Average Norm. Error', 'refineR', 'Simulated', np.mean(errors)))

    # write summary CSV
    df = pd.DataFrame(results, columns=['Metric', 'Method', 'Dataset', 'Score'])
    df['Score'] = df['Score'].round(3)
    out_path = os.path.join(_script_dir, 'summary_scores.csv')
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 50)
    print(f"Summary saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ribench_dir', type=str, default=None,
                        help='Path to RIbench Data/ directory (default: data/RIbench/Data/)')
    parser.add_argument('--meta_csv', type=str, default=None,
                        help='Path to RIbench metadata CSV (default: data/RIbench/BMTestSets_meta.csv)')
    args = parser.parse_args()
    main(ribench_dir=args.ribench_dir, meta_csv=args.meta_csv)
