import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import base64
import io
import os

def run():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    df = pd.read_csv(data_path)
    
    # Prepare numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # KMeans plot
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
    ax1.set_title('KMeans Clustering (k=3)')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    
    # DBSCAN plot
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
    ax2.set_title('DBSCAN Clustering')
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    
    plt.tight_layout()
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # Calculate metrics
    n_clusters_kmeans = len(np.unique(kmeans_labels))
    n_clusters_dbscan = len(np.unique(dbscan_labels[dbscan_labels != -1]))
    n_noise_dbscan = np.sum(dbscan_labels == -1)
    
    return {
        "experiment": "Clustering Analysis",
        "metrics": {
            "kmeans_clusters": n_clusters_kmeans,
            "dbscan_clusters": n_clusters_dbscan,
            "dbscan_noise_points": n_noise_dbscan,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "features_used": len(numeric_cols)
        },
        "plot": plot_data
    }
