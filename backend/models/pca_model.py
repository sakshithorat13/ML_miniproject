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
        
        return {
            "experiment": "Principal Component Analysis (PCA)",
            "metrics": {
                "total_features": len(feature_cols),
                "components_used": 2,
                "explained_variance_pc1": round(explained_variance_ratio[0], 4),
                "explained_variance_pc2": round(explained_variance_ratio[1], 4),
                "total_explained_variance": round(cumulative_variance[1], 4),
                "feature_names": feature_cols[:2]
            },
            "plot": plot_data
        }
    except Exception as e:
        return {
            "experiment": "Principal Component Analysis (PCA)",
            "error": str(e),
            "plot": ""
        }
