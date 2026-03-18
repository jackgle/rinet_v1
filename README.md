# RINet v1

Neural network for reference interval estimation from mixed clinical laboratory data.

## Setup

```bash
pip install -r requirements.txt
Rscript requirements.R
```

## Reproducing the experiment

### 1. Generate simulated training data

Fit a parametric model to RIbench mixture parameters, then sample synthetic training/test sets.

```bash
cd data
Rscript generateRIbench.R
python generate_samples.py --meta_csv path/to/RIbench/SpecificationTestSets.csv
```

This reads `RIbench/SpecificationTestSets.csv`, fits the `RIbenchModeler`, and outputs pickle files to `data/simulated/`.

### 2. Train the model

```bash
cd modeling
python train.py
```

Saves the trained CNN checkpoint and scaler to `modeling/model/`.

### 3. Prepare simulated test data for refineR

```bash
python evaluation/simulated/write_test_csvs.py
```

Exports the simulated test set as individual CSVs to `evaluation/simulated/test_csvs/`.

### 4. Run refineR (R)

```bash
Rscript evaluation/ribench/run_refineR.R /path/to/RIbench/Data/
cd evaluation/simulated && Rscript run_refineR.R
```

Batch-processes test data through refineR. Predictions are saved to `refineR_predictions/` in each directory. The RIbench script uses `modBoxCox` for LDH and the 95th percentile for CRP automatically.

### 5. Evaluate

```bash
python evaluation/evaluate_all.py --ribench_dir /path/to/RIbench/Data/ --meta_csv /path/to/SpecificationTestSets.csv
```

Runs all four evaluations (RINet and refineR on both RIbench and simulated data) and writes `evaluation/summary_scores.csv`.

By default, CRP and LDH are excluded. To include all analytes:

```bash
python evaluation/evaluate_all.py --ribench_dir ... --meta_csv ... --exclude none
```

#### RIbench subsampling

To run a quick evaluation on a subset of RIbench (e.g. 1000 samples), use `evaluate_rinet.py` directly:

```bash
python evaluation/ribench/evaluate_rinet.py --ribench_dir /path/to/RIbench/Data/ --meta_csv /path/to/SpecificationTestSets.csv --n_samples 1000
```

#### Plotting

To plot the summary comparison:

```bash
python evaluation/plot_summary_scores.py
```

## Citation

If you use this code in academic work, please cite the associated paper:

LeBien, J., Velev, J. & Roche-Lima, A. Indirect reference interval estimation using a convolutional neural network with application to cancer antigen 125. Sci Rep 14, 19332 (2024). https://doi.org/10.1038/s41598-024-70074-6
