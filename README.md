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
python generate_samples.py
```

This reads `RIbench/BMTestSets_meta.csv`, fits the `RIbenchModeler`, and outputs pickle files to `data/simulated/`.

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
cd evaluation/ribench && Rscript run_refineR.R
cd evaluation/simulated && Rscript run_refineR.R
```

Batch-processes test data through refineR. The RIbench script reads directly from `data/RIbench/Data/` (the full dataset). Predictions are saved to `refineR_predictions/` in each directory.

### 5. Evaluate

```bash
python evaluation/evaluate_all.py
```

Runs all four evaluations (RINet and refineR on both RIbench and simulated data) and writes `evaluation/summary_scores.csv`.

To plot the summary comparison:

```bash
python evaluation/plot_summary_scores.py
```

For per-analyte breakdowns and visualization, see the notebooks in `evaluation/`.
