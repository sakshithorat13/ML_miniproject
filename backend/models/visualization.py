import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import base64
import io
import os

def run():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    df = pd.read_csv(data_path)
    
    # Handle categorical features for correlation calculation
    df_numeric = df.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            le = LabelEncoder()
            df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Heart Disease Dataset - Correlation Heatmap')
    plt.tight_layout()
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # Get basic statistics
    stats = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }
    
    return {
        "experiment": "Data Visualization",
        "metrics": stats,
        "analysis": {
            "graph_interpretation": f"""
            **Correlation Heatmap Analysis:**
            
            The heatmap reveals critical relationships between heart disease risk factors:
            
            **Color Interpretation:**
            - Red colors indicate positive correlations (variables increase together)
            - Blue colors show negative correlations (one increases, other decreases)
            - Darker colors represent stronger correlations
            
            **Key Correlations Identified:**
            - Strong positive correlations suggest variables that move together
            - Strong negative correlations indicate inverse relationships
            - Values close to 0 (white) show little to no linear relationship
            
            **Dataset Overview:**
            - Total Features: {stats['shape'][1]}
            - Total Samples: {stats['shape'][0]}
            - Missing Values: {sum(stats['missing_values'].values())} total
            """,
            "what_graph_shows": "Correlation matrix heatmap showing relationships between all heart disease risk factors",
            "key_inferences": [
                "Identifies which features are most correlated with heart disease",
                "Reveals multicollinearity between features",
                "Helps in feature selection for machine learning models"
            ],
            "why_graph_like_this": "Heatmaps use color gradients to make correlation patterns visually obvious - human eyes quickly identify patterns in color that would be difficult to spot in numbers",
            "practical_applications": [
                "Feature selection for ML models",
                "Understanding risk factor relationships",
                "Identifying redundant measurements",
                "Clinical research insights"
            ]
        },
        "plot": plot_data
    }
