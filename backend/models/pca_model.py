# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.decomposition import PCA
# import matplotlib
# matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
# import matplotlib.pyplot as plt
# import base64
# import io
# import os

# def run():
#     try:
#         # Load dataset
#         data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#         df = pd.read_csv(data_path)
        
#         # Prepare features (exclude target column)
#         target_col = 'Heart Disease'
#         feature_cols = [col for col in df.columns if col != target_col]
        
#         X = df[feature_cols].copy()
        
#         # Handle categorical features
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))
        
#         # Fill missing values
#         X = X.fillna(X.mean())
        
#         # Standardize the data
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Apply PCA - reduce to 2D
#         pca = PCA(n_components=2)
#         X_pca = pca.fit_transform(X_scaled)
        
#         # Create before and after PCA plots
#         plt.figure(figsize=(12, 5))
        
#         # Before PCA: first two standardized features
#         plt.subplot(1, 2, 1)
#         plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', s=50, alpha=0.6)
#         plt.xlabel(f"{feature_cols[0]} (standardized)")
#         plt.ylabel(f"{feature_cols[1]} (standardized)")
#         plt.title("Before PCA: First 2 Standardized Features")
#         plt.grid(True, alpha=0.3)
        
#         # After PCA: principal components
#         plt.subplot(1, 2, 2)
#         plt.scatter(X_pca[:, 0], X_pca[:, 1], c='green', s=50, alpha=0.6)
#         plt.xlabel("Principal Component 1")
#         plt.ylabel("Principal Component 2")
#         plt.title("After PCA: 2 Principal Components")
#         plt.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         # Save plot to base64
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format='png', dpi=100)
#         buffer.seek(0)
#         plot_data = base64.b64encode(buffer.getvalue()).decode()
#         plt.close()
        
#         # Calculate explained variance
#         explained_variance_ratio = pca.explained_variance_ratio_
#         cumulative_variance = np.cumsum(explained_variance_ratio)
        
#         # Find top features by explained variance
#         top_variance_indices = np.argsort(explained_variance_ratio)[::-1][:2]
#         selected_features = [feature_cols[i] for i in top_variance_indices]
        
#         return {
#             "experiment": "Principal Component Analysis (PCA)",
#             "metrics": {
#                 "total_features": len(feature_cols),
#                 "components_used": 2,
#                 "explained_variance_pc1": round(explained_variance_ratio[0], 4),
#                 "explained_variance_pc2": round(explained_variance_ratio[1], 4),
#                 "total_explained_variance": round(cumulative_variance[1], 4),
#                 "selected_features": selected_features
#             },
#             "analysis": {
#                 "graph_interpretation": f"""
#                 **PCA Transformation Analysis:**
                
#                 **Before PCA (Left Plot):**
#                 Shows original features with highest variance - data points spread in original coordinate system
                
#                 **After PCA (Right Plot):**
#                 Shows same data transformed to principal component space - axes now represent directions of maximum variance
                
#                 **Variance Explained:**
#                 - PC1 explains {explained_variance_ratio[0]:.1%} of total variance
#                 - PC2 explains {explained_variance_ratio[1]:.1%} of total variance
#                 - Combined: {cumulative_variance[1]:.1%} of total variance captured
                
#                 **Dimensionality Reduction:**
#                 Reduced from {len(feature_cols)} features to 2 components while retaining {cumulative_variance[1]:.1%} of information
#                 """,
#                 "what_graph_shows": "Before/after comparison showing data transformation from original features to principal components",
#                 "key_inferences": [
#                     f"Two components capture {cumulative_variance[1]:.1%} of data variance",
#                     "PCA removes redundancy while preserving important patterns",
#                     "Data becomes more interpretable in reduced dimensions"
#                 ],
#                 "why_graph_like_this": "Side-by-side plots clearly show how PCA transforms data while preserving the essential structure and relationships",
#                 "practical_applications": [
#                     "Data visualization and exploration",
#                     "Feature reduction for machine learning",
#                     "Noise reduction in medical data",
#                     "Identifying key health indicators"
#                 ]
#             },
#             "plot": plot_data
#         }
#     except Exception as e:
#         return {
#             "experiment": "Principal Component Analysis (PCA)",
#             "error": str(e),
#             "plot": ""
#         }


# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # for Flask (no GUI)
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.decomposition import PCA
# import base64
# import io
# import os

# def run():
#     try:
#         # ✅ Load dataset
#         data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#         df = pd.read_csv(data_path)

#         target_col = 'Heart Disease'
#         feature_cols = [col for col in df.columns if col != target_col]
#         X = df[feature_cols].copy()

#         # ✅ Encode categorical columns
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))

#         # ✅ Handle missing values
#         X = X.fillna(X.mean())

#         # ✅ Standardize features
#         scaler = StandardScaler()
#         X_std = scaler.fit_transform(X)

#         # ✅ Apply PCA (2D)
#         pca = PCA(n_components=2)
#         X_pca = pca.fit_transform(X_std)

#         # ✅ Extract PCA components and explained variance
#         components = pca.components_
#         explained_var = pca.explained_variance_ratio_
#         mean = np.mean(X_std, axis=0)

#         # ✅ Create PCA plot with arrows (like Supriya Nayak version)
#         plt.figure(figsize=(9, 7))
#         plt.scatter(X_std[:, 0], X_std[:, 1], color='skyblue', alpha=0.5, label='Data Points')

#         # Plot principal component directions
#         for i, (comp, var) in enumerate(zip(components, explained_var)):
#             arrow_scale = var * 6.0
#             plt.arrow(0, 0,
#                       comp[0] * arrow_scale,
#                       comp[1] * arrow_scale,
#                       color='crimson',
#                       width=0.02,
#                       head_width=0.15,
#                       alpha=0.8,
#                       length_includes_head=True)
#             plt.text(comp[0] * arrow_scale * 1.2,
#                      comp[1] * arrow_scale * 1.2,
#                      f'PC{i+1} ({var*100:.1f}%)',
#                      color='darkred',
#                      fontsize=11,
#                      fontweight='bold',
#                      ha='center')

#         plt.title("PCA Visualization with Principal Directions", fontsize=14, fontweight='bold')
#         plt.xlabel(f"{feature_cols[0]} (Standardized)")
#         plt.ylabel(f"{feature_cols[1]} (Standardized)")
#         plt.grid(True, linestyle='--', alpha=0.4)
#         plt.axhline(0, color='black', linewidth=1)
#         plt.axvline(0, color='black', linewidth=1)
#         plt.axis('equal')
#         plt.legend(loc='upper right')
#         plt.tight_layout()

#         # ✅ Convert to base64
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
#         buffer.seek(0)
#         plot_data = base64.b64encode(buffer.getvalue()).decode()
#         plt.close()

#         # ✅ Return analysis info
#         cumulative_variance = np.cumsum(explained_var)
#         return {
#             "experiment": "Principal Component Analysis (PCA)",
#             "metrics": {
#                 "total_features": len(feature_cols),
#                 "components_used": 2,
#                 "explained_variance_pc1": round(explained_var[0], 4),
#                 "explained_variance_pc2": round(explained_var[1], 4),
#                 "total_explained_variance": round(cumulative_variance[1], 4),
#             },
#             "analysis": {
#                 "graph_interpretation": f"""
#                 **PCA Visualization with Principal Directions**
                
#                 - The arrows (red) show the directions of maximum variance in the dataset.
#                 - Each arrow length is proportional to the amount of variance explained.
#                 - PC1 explains {explained_var[0]:.1%} of total variance.
#                 - PC2 explains {explained_var[1]:.1%} of total variance.
#                 - Combined, they capture {cumulative_variance[1]:.1%} of total variance.
#                 """,
#                 "what_graph_shows": "Scatter of standardized features with PCA directions showing the new axes of maximum variance.",
#                 "key_inferences": [
#                     f"PCA reduces {len(feature_cols)} features to 2 while retaining {cumulative_variance[1]:.1%} information.",
#                     "Principal components help identify the most influential feature combinations.",
#                     "Arrows represent new orthogonal feature directions capturing key patterns."
#                 ],
#                 "practical_applications": [
#                     "Feature reduction for predictive modeling",
#                     "Noise filtering and data visualization",
#                     "Understanding relationships between health attributes"
#                 ]
#             },
#             "plot": plot_data
#         }

#     except Exception as e:
#         return {
#             "experiment": "Principal Component Analysis (PCA)",
#             "error": str(e),
#             "plot": ""
#         }

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # for Flask (no GUI)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import base64
import io
import os

def run():
    try:
        # ✅ Load dataset directly (use your uploaded file path)
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_disease_prediction.csv')
        df = pd.read_csv(data_path)

        # ✅ Identify target column
        target_col = 'Heart Disease' if 'Heart Disease' in df.columns else df.columns[-1]
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()

        # ✅ Encode categorical columns
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # ✅ Handle missing values
        X = X.fillna(X.mean())

        # ✅ Standardize features
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # ✅ Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

        # ✅ Identify which features are most influential in PC1 and PC2
        loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=feature_cols)
        top_pc1 = loadings['PC1'].abs().idxmax()
        top_pc2 = loadings['PC2'].abs().idxmax()

        explained_var = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_var)

        # ✅ Plot using those top 2 features
        plt.figure(figsize=(9, 7))
        plt.scatter(X[top_pc1], X[top_pc2], color='skyblue', alpha=0.6, label='Data Points')

        # ✅ Draw PCA arrows
        for i, (comp, var) in enumerate(zip(pca.components_, explained_var)):
            arrow_scale = var * 6.0
            plt.arrow(0, 0,
                      comp[0] * arrow_scale,
                      comp[1] * arrow_scale,
                      color='crimson',
                      width=0.02,
                      head_width=0.15,
                      alpha=0.8,
                      length_includes_head=True)
            plt.text(comp[0] * arrow_scale * 1.3,
                     comp[1] * arrow_scale * 1.3,
                     f'PC{i+1} ({var*100:.1f}%)',
                     color='darkred',
                     fontsize=11,
                     fontweight='bold',
                     ha='center')

        plt.title("PCA Visualization — Heart Disease Dataset", fontsize=14, fontweight='bold')
        plt.xlabel(f"{top_pc1} (Standardized)")
        plt.ylabel(f"{top_pc2} (Standardized)")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # ✅ Convert to base64 for Flask
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # ✅ Return detailed analysis
        return {
            "experiment": "Principal Component Analysis (PCA)",
            "metrics": {
                "total_features": len(feature_cols),
                "components_used": 2,
                "explained_variance_pc1": round(explained_var[0], 4),
                "explained_variance_pc2": round(explained_var[1], 4),
                "total_explained_variance": round(cumulative_variance[1], 4),
                "top_feature_pc1": top_pc1,
                "top_feature_pc2": top_pc2
            },
            "analysis": {
                "graph_interpretation": f"""
                **PCA Visualization on Heart Disease Dataset**

                - PC1 mainly depends on **{top_pc1}**
                - PC2 mainly depends on **{top_pc2}**
                - PC1 explains {explained_var[0]:.1%} of total variance
                - PC2 explains {explained_var[1]:.1%} of total variance
                - Together, they capture {cumulative_variance[1]:.1%} of variance
                """,
                "key_inferences": [
                    f"Top features influencing PCA: {top_pc1} and {top_pc2}.",
                    "The plot reveals major health indicators contributing to variance.",
                    "Helps identify key relationships influencing heart disease risk."
                ],
                "practical_applications": [
                    "Feature reduction for heart disease prediction models",
                    "Visualization of feature influence",
                    "Understanding which attributes dominate health patterns"
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
