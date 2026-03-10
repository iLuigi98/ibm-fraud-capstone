# Graph Neural Networks for Anti-Money Laundering Detection

### *Authors: Dimas Molina, Luigi Cheng, Reza Moghadam, Timothy Chang*

This project explores graph-based machine learning methods for anti-money laundering detection using transactional banking data. The workflow covers the full modeling process from exploratory data analysis and preprocessing, to feature engineering and feature selection, to model comparison and final model selection.

The final selected model is **GNN + CatBoost**, where a Graph Neural Network learns node representations from the transaction graph and CatBoost uses those learned representations together with transaction-level features for the final prediction.

## Reproducing the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

Place the AML transaction dataset inside the `data/` directory.

### 3. Run the full pipeline

```bash
python scripts/run_pipeline.py
```

The pipeline script is intended to reproduce the end-to-end modeling flow and save outputs into the `results/` folder.

## Repository Structure

```text
data/                         # Dataset files
docs/                         # Project notes and documentation
notebooks/                    # Research and modeling notebooks
  01_eda_data_preprocessing.ipynb
  02_feature_tester.ipynb
  03_edgemlp/
  04_gnn/
scripts/
  run_pipeline.py             # Reproducible end-to-end pipeline script
results/                      # Generated outputs from the pipeline
requirements.txt
README.md
```

## Project Workflow

### 1. Exploratory Data Analysis and Preprocessing

Notebook:
- `notebooks/01_eda_data_preprocessing.ipynb`

This stage focuses on understanding the data and preparing it for modeling. The work includes:

- inspecting dataset structure and transaction fields
- examining laundering label imbalance
- studying amount, currency, payment format, and timestamp patterns
- cleaning and standardizing fields
- generating reusable preprocessing outputs for downstream graph modeling

Key preprocessing steps include:

- converting transaction amounts to numeric format
- handling missing or malformed values
- transforming timestamps into usable temporal variables
- creating log-scaled amount features
- preparing graph identifiers for source and destination accounts

### 2. Feature Engineering and Feature Selection

Notebook:
- `notebooks/02_feature_tester.ipynb`

This stage evaluates multiple feature groups to identify which transaction features contribute most to validation performance.

Candidate feature groups included:

- **raw transaction features**
  - `Amount Received`
  - `Amount Paid`
  - `Log Amount Received`
  - `Receiving Currency`
  - `Payment Currency`
  - `Payment Format`

- **temporal features**
  - cyclical hour encoding with `tx_hour_sin` and `tx_hour_cos`

Feature combinations were compared using **validation PR-AUC**, which is the primary metric because money laundering detection is a highly imbalanced classification task.

The best feature-engineered set identified during feature selection was:

- `Amount Received`
- `Amount Paid`
- `Log Amount Received`
- `Receiving Currency`
- `Payment Currency`
- `Payment Format`
- `tx_hour_sin`
- `tx_hour_cos`

Categorical features such as currency and payment format are one-hot encoded before being passed into the graph model.

### 3. Model Selection

The project compares multiple model families before selecting the final approach.

#### EdgeMLP

Location:
- `notebooks/03_edgemlp/`

These models treat transactions primarily as edge-level classification problems and provide useful baselines.

#### GNN Variants

Location:
- `notebooks/04_gnn/`

The GNN experiments represent the dataset as a graph:

- **nodes** = accounts
- **edges** = transactions

This allows the model to learn structural patterns such as:

- repeated flows between accounts
- suspicious interaction neighborhoods
- graph connectivity patterns related to laundering behavior

The GNN experiments were used to compare:

- original transaction-feature GNN models
- feature-engineered GNN models
- stacked models such as **GNN + XGBoost** and **GNN + CatBoost**

### 4. Final Model

The final selected model is **GNN + CatBoost**.

This model works in two stages:

1. A Graph Neural Network is trained on the transaction graph using the selected transaction and engineered features.
2. Learned node embeddings from the GNN are combined with transaction-level information and passed into a CatBoost classifier.

This final setup was selected because it combines:

- graph structure learning from the GNN
- strong tabular classification from CatBoost
- improved performance compared with simpler baselines

## Why PR-AUC Matters

Accuracy is not appropriate for this problem because the dataset is highly imbalanced. A model could predict the majority class most of the time and still appear accurate while failing to detect laundering cases.

For that reason, the project emphasizes:

- **PR-AUC** as the main metric
- **ROC-AUC** as a secondary metric
- **confusion matrices** for final model inspection

## Outputs

Running the pipeline script should generate outputs in the `results/` folder, such as:

- saved metrics
- confusion matrix visualizations
- model result summaries