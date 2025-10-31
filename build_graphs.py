import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import joblib

# ==========================
# CONFIG
# ==========================
CSV_PATH = "UNSW_NB15.csv"   
ROWS_PER_GRAPH = 1000
LABEL_COLUMN = "label"        
K = 5                        
OUTPUT_PATH = "graphs.pt"

# ==========================
# LOAD CSV
# ==========================
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

# Ensure label exists
if LABEL_COLUMN not in df.columns:
    raise ValueError(f"Label column '{LABEL_COLUMN}' not found!")

# Separate features and labels
y = df[LABEL_COLUMN].values
X = df.drop(columns=[LABEL_COLUMN])

# Convert categorical columns if any
X = X.select_dtypes(include=['float64', 'int64']).fillna(0)

# ==========================
# BUILD GRAPHS
# ==========================
graphs = []
num_graphs = len(X) // ROWS_PER_GRAPH

for i in range(num_graphs):
    start = i * ROWS_PER_GRAPH
    end = start + ROWS_PER_GRAPH
    X_chunk = X.iloc[start:end].values
    y_chunk = y[start:end]

    # Graph Label Rule A → Attack if ANY attack row exists
    graph_label = 1 if np.any(y_chunk == 1) else 0

    # KNN for edges
    nbrs = NearestNeighbors(n_neighbors=K+1).fit(X_chunk)
    distances, indices = nbrs.kneighbors(X_chunk)

    edge_index = []
    for node_idx, neighbors in enumerate(indices):
        for n in neighbors[1:]:
            edge_index.append([node_idx, n])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x_tensor = torch.tensor(X_chunk, dtype=torch.float)
    y_tensor = torch.tensor([graph_label], dtype=torch.long)

    graph = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    graphs.append(graph)

print(f"Total graphs created: {len(graphs)}")

# Save dataset
torch.save(graphs, OUTPUT_PATH)
print(f"Saved graph dataset to {OUTPUT_PATH}")
