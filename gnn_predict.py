import torch
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import SAGEConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Allow PyTorch to load custom graph objects
torch.serialization.add_safe_globals([Data])

# ==========================
# DEFINE MODEL
# ==========================
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset & model
graphs = torch.load("graphs.pt", weights_only=False)
model = GraphSAGE(in_channels=graphs[0].x.size(1), hidden_channels=64, num_classes=2).to(device)
model.load_state_dict(torch.load("gnn_model.pth", map_location=device))
model.eval()

loader = DataLoader(graphs, batch_size=1)

print("✅ Running inference and exporting...")

# ======== GRID VISUALIZATION SETTINGS ========
GRID_SIZE = 4  # 4 graphs
rows, cols = 2, 2  # 2×2 layout
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
axes = axes.flatten()

for idx, (ax, data) in enumerate(zip(axes, loader)):
    if idx >= GRID_SIZE: break  # Only first 4

    data = data.to(device)
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1).item()
    label_text = "ATTACK" if pred == 1 else "NORMAL"

    # Export CSVs
    nodes_df = pd.DataFrame(data.x.cpu().numpy())
    nodes_df.to_csv(f"graph_{idx}_nodes.csv", index_label="node_id")
    
    edges = data.edge_index.cpu().numpy().T
    edges_df = pd.DataFrame(edges, columns=["src", "dst"])
    edges_df.to_csv(f"graph_{idx}_edges.csv", index=False)

    print(f"✅ Saved graph_{idx}_nodes.csv & graph_{idx}_edges.csv")

    # Plot graph
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)
    nx.draw(G_nx, node_size=10, with_labels=False, ax=ax)
    ax.set_title(f"Graph {idx} - {label_text}")

plt.tight_layout()
plt.savefig("graph_grid.png")
print("✅ Saved 2×2 grid as graph_grid.png")
plt.show()
