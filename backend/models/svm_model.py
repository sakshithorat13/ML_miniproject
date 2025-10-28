# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import matplotlib
# matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
# import matplotlib.pyplot as plt
# import base64
# import io
# import os

# def run():
#     # Load dataset
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#     df = pd.read_csv(data_path)
    
#     # Prepare features and target
#     target_col = 'target' if 'target' in df.columns else df.columns[-1]
#     feature_cols = [col for col in df.columns if col != target_col]
    
#     X = df[feature_cols].fillna(df[feature_cols].mean())
#     y = df[target_col]
    
#     # Convert target to numeric if it's categorical
#     if y.dtype == 'object':
#         y_numeric = pd.Categorical(y).codes
#     else:
#         y_numeric = y
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Train SVM model
#     model = SVC(kernel='rbf', random_state=42)
#     model.fit(X_train_scaled, y_train)
    
#     # Predictions
#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     # Create feature importance plot (using first two features for visualization)
#     plt.figure(figsize=(10, 6))
#     plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
#     plt.xlabel(f'Feature 1: {feature_cols[0]}')
#     plt.ylabel(f'Feature 2: {feature_cols[1]}')
#     plt.title('SVM Classification Results (2D Projection)')
#     plt.colorbar(label='Predicted Class')
#     plt.tight_layout()
    
#     # Save plot to base64
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png', dpi=100)
#     buffer.seek(0)
#     plot_data = base64.b64encode(buffer.getvalue()).decode()
#     plt.close()
    
#     return {
#         "experiment": "Support Vector Machine",
#         "metrics": {
#             "accuracy": round(accuracy, 4),
#             "kernel": "RBF",
#             "target_variable": target_col,
#             "features_used": len(feature_cols)
#         },
#         "plot": plot_data
#     }


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import matplotlib
# matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
# import matplotlib.pyplot as plt
# import base64
# import io
# import os

# def run():
#     # Load dataset
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#     df = pd.read_csv(data_path)
    
#     # Prepare features and target
#     target_col = 'target' if 'target' in df.columns else df.columns[-1]
#     feature_cols = [col for col in df.columns if col != target_col]
    
#     X = df[feature_cols].fillna(df[feature_cols].mean())
#     y = df[target_col]
    
#     # Convert target to numeric if it's categorical
#     if y.dtype == 'object':
#         y_numeric = pd.Categorical(y).codes
#     else:
#         y_numeric = y
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Train SVM model
#     model = SVC(kernel='rbf', random_state=42)
#     model.fit(X_train_scaled, y_train)
    
#     # Predictions
#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     # Create feature importance plot (using first two features for visualization)
#     plt.figure(figsize=(10, 6))
#     plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
#     plt.xlabel(f'Feature 1: {feature_cols[0]}')
#     plt.ylabel(f'Feature 2: {feature_cols[1]}')
#     plt.title('SVM Classification Results (2D Projection)')
#     plt.colorbar(label='Predicted Class')
#     plt.tight_layout()
    
#     # Save plot to base64
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png', dpi=100)
#     buffer.seek(0)
#     plot_data = base64.b64encode(buffer.getvalue()).decode()
#     plt.close()
    
#     return {
#         "experiment": "Support Vector Machine",
#         "metrics": {
#             "accuracy": round(accuracy, 4),
#             "kernel": "RBF",
#             "target_variable": target_col,
#             "features_used": len(feature_cols)
#         },
#         "plot": plot_data
#     }



import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (important for Flask)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import base64
import io
import os
import numpy as np
from typing import Optional


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for web display."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return data


def run(age: Optional[int] = None):
    """Train an SVM model and visualize its decision boundary on training and test data."""

    # ✅ Dataset path
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {data_path}")

    print(f"📁 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # --- Encode categorical columns ---
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

    # --- Select features and target ---
    if "Heart Disease" not in df_encoded.columns:
        raise KeyError("❌ Column 'Heart Disease' not found in dataset.")

    X = df_encoded[["Age", "Cholesterol"]]
    y = df_encoded["Heart Disease"]

    # --- Standardize features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # --- Train SVM ---
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # --- Performance metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # --- Create mesh grid for boundary visualization ---
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    def plot_decision_boundary(ax, X_data, y_data, title):
        """Plot decision boundary, margins, and support vectors."""
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Fill decision regions
        ax.contourf(xx, yy, Z > 0, alpha=0.3, cmap="coolwarm")

        # Draw hyperplane and margins
        ax.contour(xx, yy, Z, colors='black', levels=[-1, 0, 1],
                   linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

        # Data points
        ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data,
                   cmap="coolwarm", edgecolor='k', s=70, alpha=0.9)

        # Support vectors
        ax.scatter(svm.support_vectors_[:, 0],
                   svm.support_vectors_[:, 1],
                   s=130, facecolors='none', edgecolors='black',
                   linewidths=2, label='Support Vectors')

        ax.set_xlabel("Age (Standardized)")
        ax.set_ylabel("Cholesterol (Standardized)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="upper left")

    # --- Create plots for Training and Test sets ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plot_decision_boundary(axes[0], X_train, y_train, "SVM Decision Boundary (Training Data)")
    plot_decision_boundary(axes[1], X_test, y_test, "SVM Decision Boundary (Test Data)")
    plt.tight_layout()

    # --- Convert to base64 for web ---
    plots = {"SVM_Train_Test_Comparison": _fig_to_base64(fig)}

    # --- Insights Section (Enhanced & Detailed) ---
    insights_table = [
        {
            "Graph": "🧩 SVM Decision Boundary (Training vs Test)",
            "Insight": (
                "<b>Overview:</b> These plots visualize how the Support Vector Machine (SVM) model "
                "separates patients with and without heart disease using two key features — <b>Age</b> and <b>Cholesterol</b>.<br><br>"

                "<b>🔹 Training Plot (Top):</b><ul>"
                "<li>The <b>solid black line</b> represents the optimal hyperplane separating the two classes.</li>"
                "<li>The <b>dashed black lines</b> indicate the margins (Z = ±1), defining the decision boundaries.</li>"
                "<li><b>Hollow points</b> are <b>support vectors</b> — the most crucial samples that define the margin width and orientation.</li>"
                "<li>Red points = patients with heart disease; Blue points = without heart disease.</li>"
                "</ul>"

                "<b>🔹 Test Plot (Bottom):</b><ul>"
                "<li>Represents unseen data to test model generalization.</li>"
                "<li>The separation pattern is similar to the training plot — showing <b>strong generalization</b>.</li>"
                "<li>A few borderline misclassifications occur where red and blue points overlap, reflecting real-world complexity.</li>"
                "</ul>"

                "<b>📊 Interpretation:</b><ul>"
                f"<li><b>Accuracy:</b> {acc*100:.2f}%</li>"
                f"<li><b>Precision:</b> {prec*100:.2f}%</li>"
                f"<li><b>Recall:</b> {rec*100:.2f}%</li>"
                f"<li><b>F1-Score:</b> {f1*100:.2f}%</li>"
                f"<li><b>Support Vectors:</b> {len(svm.support_vectors_)}</li>"
                "<li>The linear SVM effectively captures the relationship between age and cholesterol in predicting heart disease.</li>"
                "<li>However, incorporating additional health metrics (like blood pressure, ECG, max heart rate) could enhance prediction accuracy.</li>"
                "</ul>"
            )
        }
    ]

    # --- Final Output ---
    return {
        "experiment": "🔍 Support Vector Machine (SVM) — Heart Disease Classification",
        "plots": plots,
        "insights_table": insights_table,
        "metrics": {
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1 Score": f"{f1*100:.2f}%",
            "Kernel": "Linear",
            "Support Vectors": len(svm.support_vectors_),
            "Features Used": "Age, Cholesterol"
        }
    }


# ---- CLI Testing ----
if __name__ == "__main__":
    result = run()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    os.makedirs(out_dir, exist_ok=True)

    for name, b64 in result["plots"].items():
        filepath = os.path.join(out_dir, f"{name}.png")
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"✅ Saved plot: {filepath}")

    print("\n📊 Insights Table:")
    for row in result["insights_table"]:
        print(f"- {row['Graph']}: {row['Insight']}")