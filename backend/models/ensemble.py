import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    # Feature importance plot
    feature_importance = rf_model.feature_importances_
    indices = np.argsort(feature_importance)[::-1][:10]  # Top 10 features
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(indices)), feature_importance[indices])
    plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=45)
    plt.title('Random Forest - Feature Importance (Top 10)')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "experiment": "Ensemble Learning",
        "metrics": {
            "random_forest_accuracy": round(rf_accuracy, 4),
            "gradient_boosting_accuracy": round(gb_accuracy, 4),
            "target_variable": target_col,
            "n_estimators": 100,
            "features_used": len(feature_cols)
        },
        "plot": plot_data
    }
