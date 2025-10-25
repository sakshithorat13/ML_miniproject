import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
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
    
    X = df[feature_cols].copy()
    y = df[target_col]
    
    # Handle categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Fill missing values
    X = X.fillna(X.mean())
    
    # Convert target to numeric if it's categorical
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y_numeric = le_target.fit_transform(y)
    else:
        y_numeric = y
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create feature importance plot (using first two features for visualization)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.xlabel(f'Feature 1: {feature_cols[0]}')
    plt.ylabel(f'Feature 2: {feature_cols[1]}')
    plt.title('KNN Classification Results (2D Projection)')
    plt.colorbar(label='Predicted Class')
    plt.tight_layout()
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "experiment": "K-Nearest Neighbors",
        "metrics": {
            "accuracy": round(accuracy, 4),
            "k": 5,
            "target_variable": target_col,
            "features_used": len(feature_cols)
        },
        "plot": plot_data
    }