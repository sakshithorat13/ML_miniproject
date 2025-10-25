import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices], color='b', alpha=0.6)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance in Random Forest')
    plt.tight_layout()
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "experiment": "Random Forest",
        "metrics": {
            "accuracy": round(accuracy, 4),
            "target_variable": target_col,
            "features_used": len(feature_cols)
        },
        "plot": plot_data
    }