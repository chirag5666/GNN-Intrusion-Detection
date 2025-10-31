import streamlit as st
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ========= STREAMLIT SETTINGS =========
st.set_page_config(page_title="GNN Intrusion Detection", layout="wide")

# ========= MODEL DEFINITION =========
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

# ========= LOAD DATA =========
@st.cache_resource
def load_graphs():
    torch.serialization.add_safe_globals([Data])
    return torch.load("graphs.pt", weights_only=False)

@st.cache_resource
def load_model(input_dim):
    device = torch.device("cpu")
    model = GraphSAGE(in_channels=input_dim, hidden_channels=64, num_classes=2).to(device)
    model.load_state_dict(torch.load("gnn_model.pth", map_location=device))
    model.eval()
    return model

graphs = load_graphs()
model = load_model(graphs[0].x.size(1))

st.title(" GNN Intrusion Detection System (GraphSAGE + Streamlit)")

# ========= GRAPH SELECTION =========
graph_index = st.slider("Select Graph Index", 0, len(graphs)-1, 0)
data = graphs[graph_index]
device = torch.device("cpu")

# ========= PREDICT =========
with torch.no_grad():
    out = model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
    pred = out.argmax(dim=1).item()

label_text = "🟥 ATTACK" if pred == 1 else "🟩 NORMAL"
st.subheader(f"Prediction: {label_text}")

# ========= VISUALIZE =========
st.write("### Graph Visualization")

G_nx = nx.Graph()
edges = data.edge_index.numpy().T
G_nx.add_edges_from(edges)

fig, ax = plt.subplots(figsize=(6,6))
nx.draw(G_nx, node_size=10, with_labels=False, ax=ax)
st.pyplot(fig)

# ========= EXPORT =========
if st.button("Export Nodes & Edges as CSV"):
    nodes_df = pd.DataFrame(data.x.numpy())
    nodes_df.to_csv(f"graph_{graph_index}_nodes.csv", index_label="node_id")
    
    edges_df = pd.DataFrame(edges, columns=["src", "dst"])
    edges_df.to_csv(f"graph_{graph_index}_edges.csv", index=False)

    st.success(f"✅ Saved graph_{graph_index}_nodes.csv and graph_{graph_index}_edges.csv")
