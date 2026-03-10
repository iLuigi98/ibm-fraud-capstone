import os
import json
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import TransformerConv
import torch.nn as nn
import torch.nn.functional as F


############################################
# CONFIG
############################################

# Change these paths when running
RESULTS_DIR = "/content/drive/MyDrive/results"
DATA_PATH = "/content/drive/MyDrive/IBM_AML/HI-Small_Trans.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################
# SEED
############################################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)


############################################
# LOAD DATA
############################################

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))


############################################
# PREPROCESS
############################################

df["Amount Received"] = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0)
df["Amount Paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0)
df["Is Laundering"] = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).astype(int)

df["Log Amount Received"] = np.log1p(df["Amount Received"].clip(lower=0))

ts = pd.to_datetime(df["Timestamp"], errors="coerce")

df["tx_hour"] = ts.dt.hour.fillna(0).astype(int)
df["tx_hour_sin"] = np.sin(2 * np.pi * df["tx_hour"] / 24.0)
df["tx_hour_cos"] = np.cos(2 * np.pi * df["tx_hour"] / 24.0)


############################################
# GRAPH CONSTRUCTION
############################################

print("Building graph...")

src_keys = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
dst_keys = df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str)

all_nodes = pd.Index(pd.concat([src_keys, dst_keys]).unique())

node_to_id = pd.Series(np.arange(len(all_nodes)), index=all_nodes)

src = node_to_id.loc[src_keys].to_numpy()
dst = node_to_id.loc[dst_keys].to_numpy()

edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

num_nodes = len(all_nodes)
E = len(df)

y = df["Is Laundering"].values
y_t = torch.tensor(y, dtype=torch.long)


############################################
# FEATURE SET (BEST FROM FEATURE SELECTION)
############################################

NUM_FEATURES = [
    "Amount Received",
    "Amount Paid",
    "Log Amount Received",
    "tx_hour_sin",
    "tx_hour_cos"
]

CAT_FEATURES = [
    "Receiving Currency",
    "Payment Currency",
    "Payment Format"
]


def build_edge_attr(df):

    num_parts = []
    cat_parts = []

    for col in NUM_FEATURES:
        vals = df[col].astype(float).values
        vmax = np.max(np.abs(vals))
        if vmax > 0:
            vals = vals / vmax
        num_parts.append(vals.reshape(-1,1))

    for col in CAT_FEATURES:
        vals = df[col].astype(str)
        uniq = vals.unique()

        codes = pd.Categorical(vals, categories=uniq).codes
        onehot = np.eye(len(uniq))[codes]

        cat_parts.append(onehot)

    parts = num_parts + cat_parts
    edge_attr = np.concatenate(parts, axis=1)

    return edge_attr.astype(np.float32)


edge_attr = build_edge_attr(df)

edge_attr_t = torch.tensor(edge_attr)

data = Data(
    num_nodes=num_nodes,
    edge_index=edge_index,
    edge_attr=edge_attr_t
)


############################################
# SPLIT
############################################

perm = torch.randperm(E)

train_size = int(0.7 * E)
val_size = int(0.15 * E)

train_idx = perm[:train_size]
val_idx = perm[train_size:train_size+val_size]
test_idx = perm[train_size+val_size:]


############################################
# GNN MODEL
############################################

class EdgeAwareGNN(nn.Module):

    def __init__(self, num_nodes, hidden, edge_dim):

        super().__init__()

        self.emb = nn.Embedding(num_nodes, hidden)

        self.conv1 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hidden, hidden, edge_dim=edge_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden*2 + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden,2)
        )


    def encode(self, node_ids, edge_index, edge_attr):

        x = self.emb(node_ids)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)

        return x


############################################
# TRAIN GNN
############################################

print("Training GNN...")

model = EdgeAwareGNN(num_nodes, 128, edge_attr.shape[1]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

criterion = nn.CrossEntropyLoss()

for epoch in range(3):

    optimizer.zero_grad()

    node_ids = torch.arange(num_nodes).to(device)

    emb = model.encode(node_ids, edge_index.to(device), edge_attr_t.to(device))

    src_h = emb[src]
    dst_h = emb[dst]

    X = torch.cat([src_h, dst_h], dim=1)

    logits = torch.randn(len(src),2).to(device)

    y_torch = torch.tensor(y).to(device)

    loss = criterion(logits, y_torch)

    loss.backward()
    optimizer.step()

    print("Epoch", epoch, "Loss", loss.item())


############################################
# BUILD CATBOOST DATASET
############################################

print("Extracting embeddings...")

node_ids = torch.arange(num_nodes).to(device)

with torch.no_grad():
    emb = model.encode(node_ids, edge_index.to(device), edge_attr_t.to(device))

emb = emb.cpu().numpy()

src_emb = emb[src]
dst_emb = emb[dst]

X = np.concatenate([src_emb, dst_emb, edge_attr], axis=1)


############################################
# TRAIN CATBOOST
############################################

print("Training CatBoost...")

X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

cat = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    verbose=False
)

cat.fit(X_train, y_train, eval_set=(X_val,y_val))


############################################
# EVALUATION
############################################

test_probs = cat.predict_proba(X_test)[:,1]

pr = average_precision_score(y_test, test_probs)
roc = roc_auc_score(y_test, test_probs)

print("Test PR-AUC:", pr)
print("Test ROC-AUC:", roc)


############################################
# CONFUSION MATRIX
############################################

y_pred = (test_probs >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.title("GNN + CatBoost Confusion Matrix")

plt.savefig(os.path.join(RESULTS_DIR,"confusion_matrix.png"))


############################################
# SAVE METRICS
############################################

metrics = {
    "test_pr_auc": float(pr),
    "test_roc_auc": float(roc),
}

with open(os.path.join(RESULTS_DIR,"metrics.json"),"w") as f:
    json.dump(metrics,f,indent=4)

print("Results saved to:", RESULTS_DIR)