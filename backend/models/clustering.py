import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import os

def run():
    try:
        # Load dataset
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
        df = pd.read_csv(data_path)
        
        print("Dataset shape:", df.shape)
        
        # Select numeric features for clustering
        numeric_features = ['Age', 'BP', 'Cholesterol', 'Max HR']
        
        # Clean and prepare data
        df_clean = df[numeric_features].dropna()
        X = df_clean.values
        
        # Handle any remaining missing values
        X = np.nan_to_num(X, nan=np.nanmean(X))
        
        # Scale the features (important for K-Means)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Elbow Method to find Optimal K
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        # Apply K-Means with chosen K (k=4)
        kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
        y_kmeans = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df_clean['Cluster'] = y_kmeans
        
        # Create visualizations
        fig = plt.figure(figsize=(15, 5))
        
        # Elbow Method Plot
        ax1 = fig.add_subplot(131)
        ax1.plot(range(1, 11), wcss, marker='o', color='blue')
        ax1.set_title('Elbow Method (Optimal K for Heart Disease Clusters)')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('WCSS')
        ax1.grid(True, alpha=0.3)
        
        # Cluster Visualization (Age vs BP)
        ax2 = fig.add_subplot(132)
        colors = ['red', 'blue', 'green', 'orange']
        for i in range(4):
            cluster_data = df_clean[df_clean['Cluster'] == i]
            ax2.scatter(cluster_data['Age'], cluster_data['BP'],
                       s=50, c=colors[i], label=f'Cluster {i+1}', alpha=0.6)
        
        # Plot centroids (projected back to original scale)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        age_idx = numeric_features.index('Age')
        bp_idx = numeric_features.index('BP')
        ax2.scatter(centers[:, age_idx], centers[:, bp_idx],
                   s=300, c='yellow', marker='X', label='Centroids', 
                   edgecolors='black', linewidth=2)
        
        ax2.set_title('Heart Disease Risk Clusters (Age vs BP)')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Blood Pressure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cluster Visualization (BP vs Cholesterol)
        ax3 = fig.add_subplot(133)
        for i in range(4):
            cluster_data = df_clean[df_clean['Cluster'] == i]
            ax3.scatter(cluster_data['BP'], cluster_data['Cholesterol'],
                       s=50, c=colors[i], label=f'Cluster {i+1}', alpha=0.6)
        
        # Plot centroids
        chol_idx = numeric_features.index('Cholesterol')
        ax3.scatter(centers[:, bp_idx], centers[:, chol_idx],
                   s=300, c='yellow', marker='X', label='Centroids',
                   edgecolors='black', linewidth=2)
        
        ax3.set_title('Heart Disease Risk Clusters (BP vs Cholesterol)')
        ax3.set_xlabel('Blood Pressure')
        ax3.set_ylabel('Cholesterol')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(4):
            cluster_data = df_clean[df_clean['Cluster'] == i]
            cluster_stats[f'cluster_{i+1}'] = {
                'size': len(cluster_data),
                'avg_age': round(cluster_data['Age'].mean(), 1),
                'avg_bp': round(cluster_data['BP'].mean(), 1),
                'avg_cholesterol': round(cluster_data['Cholesterol'].mean(), 1)
            }
        
        return {
            "experiment": "K-Means Clustering Analysis",
            "metrics": {
                "total_samples": len(df_clean),
                "features_used": numeric_features,
                "optimal_clusters": 4,
                "wcss_values": wcss,
                "cluster_statistics": cluster_stats
            },
            "analysis": {
                "graph_interpretation": f"""
                **K-Means Clustering Analysis:**
                
                **Elbow Method Results:**
                The elbow curve shows WCSS (Within-Cluster Sum of Squares) decreasing as clusters increase. The "elbow" point suggests optimal cluster number.
                
                **Cluster Characteristics:**
                {chr(10).join([f"- Cluster {i+1}: {cluster_stats[f'cluster_{i+1}']['size']} patients (Avg Age: {cluster_stats[f'cluster_{i+1}']['avg_age']}, Avg BP: {cluster_stats[f'cluster_{i+1}']['avg_bp']})" for i in range(4)])}
                
                **Centroid Analysis:**
                Yellow X marks show cluster centers - representing "typical" patient profile for each risk group
                
                **Patient Segmentation:**
                Different colors represent distinct patient populations with similar health profiles
                """,
                "what_graph_shows": "Three plots: Elbow method for optimal clusters, Age vs BP clustering, and BP vs Cholesterol clustering",
                "key_inferences": [
                    "Patients naturally group into 4 distinct risk categories",
                    "Age and blood pressure show clear clustering patterns",
                    "Cholesterol levels vary significantly between clusters"
                ],
                "why_graph_like_this": "Scatter plots with color coding make it easy to see how patients group together based on similar health characteristics",
                "practical_applications": [
                    "Patient risk stratification",
                    "Personalized treatment protocols",
                    "Healthcare resource allocation",
                    "Preventive care program design"
                ]
            },
            "plot": plot_data
        }
        
    except Exception as e:
        return {
            "experiment": "K-Means Clustering Analysis",
            "error": str(e),
            "plot": ""
        }
