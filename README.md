# 🧠 Graph Neural Network (GraphSAGE) Based Intrusion Detection System

### 🔒 Project Overview
This project presents an **Intrusion Detection System (IDS)** built using **Graph Neural Networks (GNN)** — specifically the **GraphSAGE** architecture — to identify malicious patterns in network traffic data.  
The model is trained on the **UNSW-NB15 dataset**, one of the most comprehensive modern intrusion datasets, and classifies each network flow as either **Normal** or **Attack**.

Unlike traditional deep learning models that process each record independently, this system represents traffic as **graphs**, where:
- **Nodes** represent network flows (connections)
- **Edges** represent similarity between flows (using K-Nearest Neighbors)

By capturing these relational dependencies, the GraphSAGE model learns how network behaviors interact, leading to improved accuracy and robustness against complex and evolving attacks.

---

## 📊 Key Highlights
- **Dataset Used:** UNSW-NB15 (by Australian Centre for Cyber Security)
- **Model Type:** Graph Neural Network (GraphSAGE)
- **Goal:** Detect and classify network intrusions in traffic flows
- **Accuracy Achieved:** ≈ **95.6%**
- **Visualization:** Graph plots showing Normal vs Attack connections
- **Frameworks:** PyTorch, PyTorch Geometric, Streamlit

---

## 🧰 Tech Stack
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python 3.10 |
| Deep Learning Framework | PyTorch |
| Graph Processing | PyTorch Geometric |
| Visualization | NetworkX, Matplotlib |
| Interface | Streamlit |
| Data Handling | Pandas, Scikit-learn |

---

## 📂 Project Structure
