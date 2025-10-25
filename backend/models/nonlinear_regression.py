import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import base64
import io
import os

def run():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    df = pd.read_csv(data_path)
    
    # Select numeric columns for regression
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = 'chol' if 'chol' in numeric_cols else numeric_cols[0]
    feature_cols = [col for col in numeric_cols if col != target_col][:3]  # Use first 3 features
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply polynomial features
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Train polynomial regression
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_poly)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create prediction vs actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Polynomial Regression (Degree 2): Actual vs Predicted ({target_col})')
    plt.tight_layout()
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "experiment": "Multivariate Nonlinear Regression",
        "metrics": {
            "polynomial_degree": 2,
            "mse": round(mse, 4),
            "r2_score": round(r2, 4),
            "target_variable": target_col,
            "features_used": len(feature_cols)
        },
        "plot": plot_data
    }
