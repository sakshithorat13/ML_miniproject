import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import base64
import io
import os

def run():
    try:
        # Load dataset
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
        df = pd.read_csv(data_path)
        
        # Prepare features (exclude target column)
        target_col = 'Heart Disease'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].copy()
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA - reduce to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create before and after PCA plots
        plt.figure(figsize=(12, 5))
        
        # Before PCA: first two standardized features
        plt.subplot(1, 2, 1)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', s=50, alpha=0.6)
        plt.xlabel(f"{feature_cols[0]} (standardized)")
        plt.ylabel(f"{feature_cols[1]} (standardized)")
        plt.title("Before PCA: First 2 Standardized Features")
        plt.grid(True, alpha=0.3)
        
        # After PCA: principal components
        plt.subplot(1, 2, 2)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='green', s=50, alpha=0.6)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("After PCA: 2 Principal Components")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find top features by explained variance
        top_variance_indices = np.argsort(explained_variance_ratio)[::-1][:2]
        selected_features = [feature_cols[i] for i in top_variance_indices]
        
        return {
            "experiment": "Principal Component Analysis (PCA)",
            "metrics": {
                "total_features": len(feature_cols),
                "components_used": 2,
                "explained_variance_pc1": round(explained_variance_ratio[0], 4),
                "explained_variance_pc2": round(explained_variance_ratio[1], 4),
                "total_explained_variance": round(cumulative_variance[1], 4),
                "selected_features": selected_features
            },
            "analysis": {
                "graph_interpretation": f"""
                **PCA Transformation Analysis:**
                
                **Before PCA (Left Plot):**
                Shows original features with highest variance - data points spread in original coordinate system
                
                **After PCA (Right Plot):**
                Shows same data transformed to principal component space - axes now represent directions of maximum variance
                
                **Variance Explained:**
                - PC1 explains {explained_variance_ratio[0]:.1%} of total variance
                - PC2 explains {explained_variance_ratio[1]:.1%} of total variance
                - Combined: {cumulative_variance[1]:.1%} of total variance captured
                
                **Dimensionality Reduction:**
                Reduced from {len(feature_cols)} features to 2 components while retaining {cumulative_variance[1]:.1%} of information
                """,
                "what_graph_shows": "Before/after comparison showing data transformation from original features to principal components",
                "key_inferences": [
                    f"Two components capture {cumulative_variance[1]:.1%} of data variance",
                    "PCA removes redundancy while preserving important patterns",
                    "Data becomes more interpretable in reduced dimensions"
                ],
                "why_graph_like_this": "Side-by-side plots clearly show how PCA transforms data while preserving the essential structure and relationships",
                "practical_applications": [
                    "Data visualization and exploration",
                    "Feature reduction for machine learning",
                    "Noise reduction in medical data",
                    "Identifying key health indicators"
                ]
            },
            "plot": plot_data
        }
    except Exception as e:
        return {
            "experiment": "Principal Component Analysis (PCA)",
            "error": str(e),
            "plot": ""
        }
