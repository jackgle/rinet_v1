"""Export simulated test data as individual CSVs for refineR processing."""
import os
import shutil
import pickle
import pandas as pd


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'simulated')
    outdir = os.path.join(os.path.dirname(__file__), 'test_csvs')

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    # load data
    test_x = pickle.load(open(os.path.join(data_path, 'x_test.pkl'), 'rb'))

    for i in range(len(test_x)):
        if i % 100 == 0:
            print(f"{i}/{len(test_x)}")
        sample = test_x[i] - min(test_x[i])
        pd.DataFrame(sample).to_csv(os.path.join(outdir, f'{i}.csv'), index=None, header=None)

    print(f"Wrote {len(test_x)} CSVs to {outdir}")


if __name__ == '__main__':
    main()
