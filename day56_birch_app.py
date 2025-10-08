import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import numpy as np

# Streamlit app setup
st.set_page_config(page_title="Birch Clustering", page_icon="🌳", layout="centered")

st.title("🌳 Birch Clustering")
st.write("""
Birch (Balanced Iterative Reducing and Clustering using Hierarchies) is a scalable clustering algorithm 
that builds a tree structure to efficiently cluster large datasets.
""")

# Sidebar for user controls
st.sidebar.header("Birch Hyperparameters")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
threshold = st.sidebar.slider("Threshold", 0.1, 2.0, 0.5)
branching_factor = st.sidebar.slider("Branching Factor", 10, 100, 50)

# Generate synthetic data
X, _ = make_blobs(n_samples=500, centers=n_clusters, cluster_std=1.0, random_state=42)

# Train Birch model
model = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)
y_pred = model.fit_predict(X)

# Plot results
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', s=50, alpha=0.7)
ax.set_title("Birch Clustering Results")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.colorbar(scatter)
st.pyplot(fig)

# Display cluster information
unique_clusters = np.unique(y_pred)
st.subheader("📊 Cluster Summary")
st.write(f"Number of clusters found: **{len(unique_clusters)}**")
st.write("Cluster IDs:", unique_clusters.tolist())

# Optional - CF Subcluster count
st.subheader("🌲 CF Subclusters Information")
st.write(f"Number of subclusters formed: **{len(model.subcluster_centers_)}**")
