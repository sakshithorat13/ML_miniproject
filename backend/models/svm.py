import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import base64
import io
import os

def run():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM model
    model = SVC(kernel='rbf', random_state=42, probability=True)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create feature importance plot (using first two features for visualization)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.xlabel(f'Feature 1: {feature_cols[0]}')
    plt.ylabel(f'Feature 2: {feature_cols[1]}')
    plt.title('SVM Classification Results (2D Projection)')
    plt.colorbar(label='Predicted Class')
    plt.tight_layout()
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "experiment": "Support Vector Machine",
        "metrics": {
            "accuracy": round(accuracy, 4),
            "kernel": "RBF",
            "target_variable": target_col,
            "features_used": len(feature_cols)
        },
        "plot": plot_data
    }
