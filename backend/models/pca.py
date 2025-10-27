# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import base64
# import io
# import os

# def run():
#     # Load dataset
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#     df = pd.read_csv(data_path)
    
#     # Prepare numeric features
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Apply PCA
#     pca = PCA()
#     X_pca = pca.fit_transform(X_scaled)
    
#     # Apply SVD
#     svd = TruncatedSVD(n_components=min(10, X.shape[1]))
#     X_svd = svd.fit_transform(X_scaled)
    
#     # Create plots
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
#     # PCA explained variance
#     cumsum_var = np.cumsum(pca.explained_variance_ratio_)
#     ax1.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
#     ax1.set_xlabel('Number of Components')
#     ax1.set_ylabel('Cumulative Explained Variance')
#     ax1.set_title('PCA - Cumulative Explained Variance')
#     ax1.grid(True)
    
#     # Individual component variance
#     ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
#     ax2.set_xlabel('Principal Component')
#     ax2.set_ylabel('Explained Variance Ratio')
#     ax2.set_title('PCA - Individual Component Variance')
    
#     # 2D PCA projection
#     target_col = 'target' if 'target' in df.columns else None
#     if target_col and target_col in df.columns:
#         colors = df[target_col]
#         scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis', alpha=0.6)
#         plt.colorbar(scatter, ax=ax3)
#     else:
#         ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
#     ax3.set_xlabel('First Principal Component')
#     ax3.set_ylabel('Second Principal Component')
#     ax3.set_title('PCA - 2D Projection')
    
#     # SVD explained variance
#     ax4.bar(range(1, len(svd.explained_variance_ratio_) + 1), svd.explained_variance_ratio_)
#     ax4.set_xlabel('SVD Component')
#     ax4.set_ylabel('Explained Variance Ratio')
#     ax4.set_title('SVD - Component Variance')
    
#     plt.tight_layout()
    
#     # Save plot to base64
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png', dpi=100)
#     buffer.seek(0)
#     plot_data = base64.b64encode(buffer.getvalue()).decode()
#     plt.close()
    
#     # Calculate metrics
#     variance_95 = np.argmax(cumsum_var >= 0.95) + 1
#     variance_99 = np.argmax(cumsum_var >= 0.99) + 1
    
#     return {
#         "experiment": "PCA / SVD Dimensionality Reduction",
#         "metrics": {
#             "original_dimensions": X.shape[1],
#             "components_for_95_variance": variance_95,
#             "components_for_99_variance": variance_99,
#             "total_explained_variance": round(pca.explained_variance_ratio_.sum(), 4),
#             "first_component_variance": round(pca.explained_variance_ratio_[0], 4),
#             "second_component_variance": round(pca.explained_variance_ratio_[1], 4)
#         },
#         "plot": plot_data
#     }

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
import base64

def run():
    try:
        # ------------------------
        # 1. Load dataset
        # ------------------------
        # Replace with your CSV path or data source
        df = pd.read_csv("data/Heart_Disease_Prediction.csv")  
        
        # Select numeric columns only
        X = df.select_dtypes(include='number')
        
        # ------------------------
        # 2. Apply PCA
        # ------------------------
        pca = PCA()
        X_pca = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
        
        # ------------------------
        # 3. Metric Cards
        # ------------------------
        metrics = {
            "components_for_95_variance": int((explained_var.cumsum() >= 0.95).argmax() + 1),
            "components_for_99_variance": int((explained_var.cumsum() >= 0.99).argmax() + 1),
            "first_component_variance": float(explained_var[0]),
            "second_component_variance": float(explained_var[1]),
            "total_explained_variance": float(explained_var.sum()),
            "original_dimensions": X.shape[1]
        }

        # ------------------------
        # 4. Combined Plot (1 image)
        # ------------------------
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Cumulative Explained Variance
        axs[0].plot(explained_var.cumsum(), marker='o', color='blue')
        axs[0].set_title("Cumulative Explained Variance")
        axs[0].set_xlabel("Number of Components")
        axs[0].set_ylabel("Cumulative Variance")
        axs[0].grid(True)
        
        # Right: 2D Projection
        axs[1].scatter(X_pca[:, 0], X_pca[:, 1], color='red')
        axs[1].set_title("PCA 2D Projection")
        axs[1].set_xlabel("PC1")
        axs[1].set_ylabel("PC2")
        
        # Save figure to base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        combined_plot = base64.b64encode(buf.read()).decode('utf-8')

        # ------------------------
        # 5. Return single output
        # ------------------------
        return {
            "metrics": metrics,
            "plot": combined_plot
        }

    except Exception as e:
        return {"error": str(e)}
