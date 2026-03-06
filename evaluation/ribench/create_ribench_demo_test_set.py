"""Create a 1000-sample subset of RIbench data for evaluation."""
import os
import pickle
import numpy as np
import pandas as pd


def main():
    datapath_ribench = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'RIbench')
    out_dir = os.path.join(os.path.dirname(__file__), 'ribench_sample')
    test_set_meta = pd.read_csv(os.path.join(datapath_ribench, 'BMTestSets_meta.csv'), index_col=0)

    # collect all data files, excluding CRP and LDH
    files = []
    data_dir = os.path.join(datapath_ribench, 'Data')
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    files = [i for i in files if i.split('/')[-2] not in ['CRP', 'LDH']]

    # sample 1000
    idx_sub = np.random.choice(len(files), 1000)
    files_sub = [files[i] for i in idx_sub]

    def get_y(file):
        fileid = int(file.split('/')[-1].split('_')[0])
        target = (
            test_set_meta[test_set_meta.Index == fileid].GT_LRL.values[0],
            test_set_meta[test_set_meta.Index == fileid].GT_URL.values[0]
        )
        return np.array(target)

    # load samples and targets
    samples = []
    targets = []
    for c, i in enumerate(files_sub):
        if c % 100 == 0:
            print(f"{c}/{len(files_sub)}")
        sample = pd.read_csv(i, header=None).values.squeeze()
        samples.append(sample)
        targets.append(get_y(i))

    # save
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'x.pkl'), 'wb') as f:
        pickle.dump(samples, f)
    with open(os.path.join(out_dir, 'y.pkl'), 'wb') as f:
        pickle.dump(targets, f)
    with open(os.path.join(out_dir, 'files.pkl'), 'wb') as f:
        pickle.dump(files_sub, f)
    pd.DataFrame(files_sub, columns=['file'], index=None).to_csv(os.path.join(out_dir, 'files.csv'))

    print(f"Saved {len(samples)} samples to {out_dir}")


if __name__ == '__main__':
    main()
