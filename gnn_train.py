import torch
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# PATCH: Allow PyTorch to load custom Data class
# ---------------------------------------------------
torch.serialization.add_safe_globals([Data])

# ==========================
# LOAD GRAPH DATASET
# ==========================
graphs = torch.load("graphs.pt", weights_only=False)
print(f"Total graphs loaded: {len(graphs)}")

train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=8)

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
sample_graph = graphs[0]
model = GraphSAGE(in_channels=sample_graph.x.size(1), hidden_channels=64, num_classes=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# ==========================
# TRAIN LOOP
# ==========================
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total

for epoch in range(1, 21):
    loss = train()
    acc = test(test_loader)
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

torch.save(model.state_dict(), "gnn_model.pth")
print("✅ Model saved as gnn_model.pth")
