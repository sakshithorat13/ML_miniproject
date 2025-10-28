# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import matplotlib.pyplot as plt
# import base64
# import io
# import os

# def run():
#     # Load dataset
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#     df = pd.read_csv(data_path)
    
#     # Select numeric columns for regression
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
    
#     # Use cholesterol as target if available, otherwise use the first numeric column
#     target_col = 'chol' if 'chol' in numeric_cols else numeric_cols[0]
#     feature_cols = [col for col in numeric_cols if col != target_col]
    
#     X = df[feature_cols].fillna(df[feature_cols].mean())
#     y = df[target_col].fillna(df[target_col].mean())
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # Predictions
#     y_pred = model.predict(X_test)
    
#     # Metrics
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
    
#     # Create prediction vs actual plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_pred, alpha=0.6)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
#     plt.xlabel('Actual Values')
#     plt.ylabel('Predicted Values')
#     plt.title(f'Linear Regression: Actual vs Predicted ({target_col})')
#     plt.tight_layout()
    
#     # Save plot to base64
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png', dpi=100)
#     buffer.seek(0)
#     plot_data = base64.b64encode(buffer.getvalue()).decode()
#     plt.close()
    
#     return {
#         "experiment": "Linear Regression",
#         "metrics": {
#             "target_variable": target_col,
#             "mse": round(mse, 4),
#             "r2_score": round(r2, 4),
#             "mae": round(mae, 4),
#             "features_used": len(feature_cols)
#         },
#         "plot": plot_data
#     }

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import base64
import io
import os

def run():
    # --- Load Dataset ---
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    df = pd.read_csv(data_path)

    # --- Clean column names ---
    df.columns = df.columns.str.strip().str.lower()

    # --- Define target and features ---
    # We will predict Max HR (max heart rate)
    target_col = 'max hr' if 'max hr' in df.columns else 'max_hr'

    # Choose meaningful features (excluding Age & Cholesterol)
    possible_features = [
        'bp', 'fbs over 1', 'ekg results', 'exercise angina',
        'st depression', 'slope of st', 'number of vessels fluro', 'thallium'
    ]

    # Keep only those actually present in the dataset
    feature_cols = [col for col in possible_features if col in df.columns]

    # Ensure numeric only
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(df[feature_cols].mean())
    y = df[target_col].apply(pd.to_numeric, errors='coerce').fillna(df[target_col].mean())

    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train Model ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Predictions ---
    y_pred = model.predict(X_test)

    # --- Metrics ---
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # --- Visualization: Actual vs Predicted ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Max Heart Rate')
    plt.ylabel('Predicted Max Heart Rate')
    plt.title('Linear Regression: Actual vs Predicted Max HR')
    plt.grid(alpha=0.3)

    # Save plot as base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # --- Insights and Interpretation ---
    insights = (
        "<b>Insights and Interpretation:</b><br>"
        "This linear regression model predicts the <b>maximum heart rate (Max HR)</b> "
        "based on several cardiovascular indicators like blood pressure, ST depression, "
        "exercise-induced angina, and ECG results.<br><br>"
        "A strong positive correlation indicates that as certain health indicators improve, "
        "the expected Max HR increases. Conversely, higher ST depression or presence of exercise angina "
        "tends to lower predicted Max HR.<br><br>"
        "This model helps identify patients with potentially abnormal heart performance "
        "based on multiple measurable health attributes.<br><br>"
        f"<b>Model Performance:</b><br>"
        f"R² Score = {r2:.3f}<br>"
        f"Mean Absolute Error (MAE) = {mae:.2f}<br>"
        f"Mean Squared Error (MSE) = {mse:.2f}"
    )

    # --- Return Results ---
    return {
        "experiment": "Linear Regression (Max Heart Rate Prediction)",
        "metrics": {
            "Target Variable": target_col,
            "Features Used": ['blood pressure, ST depression, exercise-induced angina,ECG'],
            "MSE": round(mse, 4),
            "MAE": round(mae, 4),
            "R² Score": round(r2, 4)
        },
        "plot": plot_data,
        "insights_table": [
            {
                "Graph": "Linear Regression — Max Heart Rate",
                "Insight": insights
            }
        ]
    }
