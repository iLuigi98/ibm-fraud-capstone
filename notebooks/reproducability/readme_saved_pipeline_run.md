# README — Running the Saved Pipeline Notebook

This submission includes a notebook that **loads the saved trained artifacts** and **reproduces the final GNN + CatBoost test results without retraining**.

## Files used

The load-and-run notebook expects the following saved files to already exist in Google Drive under:

`/content/drive/MyDrive/saved_models/full_pipeline`

Required files:

- `catboost_model.cbm`
- `final_gnn_state_dict.pt`
- `isotonic_calibrator.pkl`
- `H_final_embeddings.npy`
- `metadata.json`
- `edge_index.npy`
- `edge_attr_t.npy`
- `y_edge_t.npy`
- `val_ids.npy`
- `test_ids.npy`

## Dataset path

The notebook reads the transaction CSV from:

`/content/drive/MyDrive/dataset/HI-Small_Trans.csv`

If the file is stored somewhere else, update the path in the notebook before running.

## How to run the notebook

1. Open the notebook `Load_model_and_Run.ipynb` in Colab.
2. Run the first cell to mount Google Drive.
3. Make sure the saved artifact files listed above are present in the expected Drive folder.
4. Run the notebook cells from top to bottom in order.

The notebook will:

- load the saved artifacts
- load the saved graph tensors and saved split IDs
- rebuild the CatBoost tabular features from the saved GNN embeddings
- run the saved CatBoost model on the validation and test sets
- reproduce the final evaluation metrics and confusion matrix

## Expected output

The notebook reproduces the final **test-set** results for the saved pipeline, including:

- validation PR-AUC and ROC-AUC
- test PR-AUC and ROC-AUC
- test confusion matrix
- test classification report
- final one-row summary table for the saved GNN + CatBoost pipeline

## Important note

This notebook is an **inference-only** notebook. It does **not retrain** the GNN or CatBoost model.

It is intended to demonstrate that the saved trained pipeline can be loaded and evaluated directly.

## If a path error occurs

Check these two paths in the notebook:

- dataset path
- saved artifact folder path

They must match the actual locations in Google Drive.

## Summary

To reproduce the saved model results:

- mount Drive
- confirm the saved artifact files are present
- run the notebook from top to bottom

That should reproduce the final saved pipeline evaluation without retraining.

