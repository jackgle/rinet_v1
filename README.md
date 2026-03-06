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

### 3. Create RIbench test subset

```bash
python evaluation/ribench/create_ribench_demo_test_set.py
```

Samples 1000 files from RIbench and saves them as pickles to `evaluation/ribench/ribench_sample/`.

### 4. Prepare simulated test data for refineR

```bash
python evaluation/simulated/write_test_csvs.py
```

Exports the simulated test set as individual CSVs to `evaluation/simulated/test_csvs/`.

### 5. Run refineR (R)

```bash
cd evaluation/ribench && Rscript run_refineR.R
cd evaluation/simulated && Rscript run_refineR.R
```

Batch-processes test data through refineR. Predictions are saved to `refineR_predictions/` in each directory.

### 6. Evaluate

```bash
python evaluation/ribench/evaluate_rinet.py
python evaluation/simulated/evaluate_rinet.py
python evaluation/ribench/evaluate_refineR.py
python evaluation/simulated/evaluate_refineR.py
```

For visualization and per-analyte breakdowns, see the corresponding notebooks in `evaluation/`.
