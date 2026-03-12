"""
Generate synthetic training and test data by fitting RIbench mixture parameters
and sampling from the fitted model.

Usage:
    python generate_samples.py [--output_dir ./simulated/]
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
import scipy.stats

from utils import (
    RIbenchModeler, create_mixture_sample,
    load_ribench_params, compute_ref_stats, compute_pathological_stats
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./simulated/')
    parser.add_argument('--n_train', type=int, default=9000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--meta_csv', type=str, default='./RIbench/SpecificationTestSets.csv')
    return parser.parse_args()


def main():
    args = parse_args()

    # load and process RIbench parameters
    params = load_ribench_params(args.meta_csv)
    ref_stats = compute_ref_stats(params)
    ref_mean = ref_stats['ref_mean']
    ref_std = ref_stats['ref_std']
    ref_skew = ref_stats['ref_skew']

    p_mean, p_std = compute_pathological_stats(params, ref_mean, ref_std)

    # compute bg_range
    bg_range = []
    for c, i in enumerate(params.iterrows()):
        i = i[1]
        bg_range.append([
            (i['bg_min'] - ref_mean[c]) / ref_std[c],
            (i['bg_max'] - ref_mean[c]) / ref_std[c]
        ])

    # sort by skew
    sort_idx = np.argsort(ref_skew)
    ref_skew_sorted = np.sort(np.array(ref_skew))
    left_mean_abs = np.abs([i[0] for i in p_mean])[sort_idx]
    right_mean = np.array([i[1] for i in p_mean])[sort_idx]
    left_std = np.array([i[0] for i in p_std])[sort_idx]
    right_std = np.array([i[1] for i in p_std])[sort_idx]
    left_edge_abs = np.abs(ref_stats['ref_zero'])[sort_idx]
    bg_max = np.array([i[1] for i in bg_range])[sort_idx]

    # fit modeler
    bin_shift = 1
    max_skew = 10
    modeler = RIbenchModeler(bin_shift, max_skew)
    modeler.fit(
        ref_skew_sorted, left_mean_abs, right_mean, left_std, right_std,
        left_edge_abs, bg_max, plot=False
    )

    # save modeler
    os.makedirs(args.output_dir, exist_ok=True)
    with open('./modeler.pkl', 'wb') as f:
        pickle.dump(modeler, f)

    # define training skew distributions
    skews_mean = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 2.00, 3.00, 5.00, 7.00, 9.00])
    skews_std = np.linspace(0.1, 1.0, len(skews_mean))
    skew_probs = np.array([2.5, 2, 1, 1, 1, 1, 1, 1, 1, 1])
    skew_probs /= sum(skew_probs)

    # validation skews
    val_skews_mean = (skews_mean[:-1] + skews_mean[1:]) / 2
    val_skews_std = (skews_std[:-1] + skews_std[1:]) / 2
    val_skew_probs = (skew_probs[:-1] + skew_probs[1:]) / 2
    val_skew_probs /= sum(val_skew_probs)

    # generate training data
    print(f"Generating {args.n_train} training samples...")
    training_x, training_y, training_sizes = [], [], []
    for i in range(args.n_train):
        if i % 1000 == 0:
            print(f"  {i}/{args.n_train}")
        idx = np.random.choice(len(skews_mean), p=skew_probs)
        skew = np.maximum(0, np.random.normal(skews_mean[idx], skews_std[idx]))
        sample_params = modeler.generate(ref_skew=skew)
        mixture, target, comp_sizes = create_mixture_sample(
            sample_params,
            sample_size=np.random.choice([500, 1000, 5000, 10000]),
            p_frac=np.random.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            p_ratio=np.random.uniform(0.1, 0.9),
            quantize_step=np.random.choice([0, 0.05, 0.1, 0.2])
        )
        training_x.append(mixture)
        training_y.append(target)
        training_sizes.append(comp_sizes)

    with open(os.path.join(args.output_dir, 'x_train.pkl'), 'wb') as f:
        pickle.dump(training_x, f)
    with open(os.path.join(args.output_dir, 'y_train.pkl'), 'wb') as f:
        pickle.dump(training_y, f)
    with open(os.path.join(args.output_dir, 'sizes_train.pkl'), 'wb') as f:
        pickle.dump(training_sizes, f)
    del training_x, training_y, training_sizes

    # generate test data
    print(f"Generating {args.n_test} test samples...")
    testing_x, testing_y, testing_sizes = [], [], []
    for i in range(args.n_test):
        if i % 1000 == 0:
            print(f"  {i}/{args.n_test}")
        idx = np.random.choice(len(val_skews_mean), p=val_skew_probs)
        skew = np.maximum(0, np.random.normal(val_skews_mean[idx], val_skews_std[idx]))
        sample_params = modeler.generate(ref_skew=skew)
        mixture, target, comp_sizes = create_mixture_sample(
            sample_params,
            sample_size=np.random.choice([500, 1000, 5000, 10000]),
            p_frac=np.random.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            p_ratio=np.random.uniform(0.1, 0.9),
            quantize_step=np.random.choice([0, 0.05, 0.1, 0.2])
        )
        testing_x.append(mixture)
        testing_y.append(target)
        testing_sizes.append(comp_sizes)

    with open(os.path.join(args.output_dir, 'x_test.pkl'), 'wb') as f:
        pickle.dump(testing_x, f)
    with open(os.path.join(args.output_dir, 'y_test.pkl'), 'wb') as f:
        pickle.dump(testing_y, f)
    with open(os.path.join(args.output_dir, 'sizes_test.pkl'), 'wb') as f:
        pickle.dump(testing_sizes, f)

    print("Done.")


if __name__ == '__main__':
    main()
