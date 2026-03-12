# README — Running the Saved Pipeline Notebook

This submission includes a notebook that loads the saved trained artifacts and reproduces the final GNN + CatBoost test results without retraining.

## What this notebook does

The notebook loads a previously trained GNN + CatBoost pipeline and evaluates it on the saved validation and test splits. It does not fit a new model. Instead, it:

- loads the saved trained artifacts
- loads the saved graph tensors and saved split IDs
- rebuilds the CatBoost tabular features from the saved GNN embeddings
- runs the saved CatBoost model on the validation and test sets
- reproduces the final evaluation metrics and confusion matrix

## Files used

The load-and-run notebook expects the following saved files to already exist in the saved artifact folder.

### Required files

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

## Why this notebook only works on HI-Small_Trans.csv dataset

This notebook is tied to the specific dataset used when the original model was trained, and it is not a general-purpose prediction notebook for arbitrary new datasets.

It only works correctly on this dataset because the saved artifacts were created from this exact graph and these exact edge records. In particular:

- the saved node embeddings in `H_final_embeddings.npy` were learned from this dataset’s graph structure
- the saved arrays such as `edge_index.npy`, `edge_attr_t.npy`, and `y_edge_t.npy` correspond to this dataset only
- the saved split files `val_ids.npy` and `test_ids.npy` are indices into these exact saved edges
- the CatBoost input features are rebuilt from these saved graph embeddings and edge relationships, so they depend on the same node IDs, edge ordering, and feature format used during training

If a different transaction dataset is used, the node indexing, edge structure, feature layout, and split indices will no longer match the saved artifacts. Because of that, the notebook would either fail due to shape/index mismatches or produce meaningless results.

So, this notebook should be understood as a saved-model reproduction notebook for the original dataset, not as a notebook for evaluating the model on new external data.

## How to run the notebook in Colab

### Colab-required cells

These cells are needed when running in Google Colab:

1. Package installation cell
   - installs the required libraries
2. Google Drive mount cell
   - mounts Drive so the notebook can access the dataset and saved artifacts
3. Path definition cells
   - these should point to the Google Drive locations for:
     - the saved artifact folder

After that, run the rest of the notebook from top to bottom in order.

### Colab path settings

For Colab, the notebook expects:

Saved artifact folder:

`/content/drive/MyDrive/saved_models/full_pipeline`

## How to run the notebook locally

### Local-run required changes

When running locally:

1. Skip the Google Drive mount cell
2. Run the package installation cell only if your local environment still needs those packages
3. Restart the kernel
4. Update the path cells so they point to your local folder locations
5. Run the remaining notebook cells from top to bottom

### Local path settings

For local use, replace the Google Drive paths with local paths.

For example:

```python
SAVE_DIR = Path"."
```

## Expected output

The notebook reproduces the final test-set results for the saved pipeline, including:

- validation PR-AUC and ROC-AUC
- test PR-AUC and ROC-AUC
- test confusion matrix
- test classification report
- final one-row summary table for the saved GNN + CatBoost pipeline

## Important note

This notebook is an inference-only notebook. It does not retrain the GNN or CatBoost model.

It is intended to demonstrate that the saved trained pipeline can be loaded and evaluated directly on the original saved dataset artifacts.

## Summary

To reproduce the saved model results:

- make sure the required files are present
- use the correct paths for your environment
- run the notebook from top to bottom
- do not use a different dataset with these saved artifacts

That should reproduce the final saved pipeline evaluation without retraining.
