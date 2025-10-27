import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os

def run():
    try:
        # Load the dataset
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
        df = pd.read_csv(data_path)
        
        # Data Preprocessing
        target_col = 'Heart Disease'
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Separate features (X) and target (y)
        X = df[feature_cols].copy()
        y = df[target_col]
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Convert target to binary if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y_binary = le_target.fit_transform(y)
        else:
            y_binary = y
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Implement Bagging (Random Forest)
        print("Training Random Forest Model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        # Implement Boosting (Gradient Boosting)
        print("Training Gradient Boosting Model...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        
        # Calculate accuracies
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        accuracy_gb = accuracy_score(y_test, y_pred_gb)
        
        # Create confusion matrices
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        cm_gb = confusion_matrix(y_test, y_pred_gb)
        
        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Random Forest Confusion Matrix
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churned', 'Churned'], 
                   yticklabels=['Not Churned', 'Churned'], ax=ax1)
        ax1.set_title('Random Forest Confusion Matrix')
        ax1.set_ylabel('Actual')
        ax1.set_xlabel('Predicted')
        
        # Gradient Boosting Confusion Matrix
        sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Not Churned', 'Churned'], 
                   yticklabels=['Not Churned', 'Churned'], ax=ax2)
        ax2.set_title('Gradient Boosting Confusion Matrix')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')
        
        plt.tight_layout()
        
        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Get feature importance from Random Forest
        feature_importance = rf_model.feature_importances_
        top_features = sorted(zip(feature_cols, feature_importance), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "experiment": "Ensemble Methods (Bagging vs Boosting)",
            "metrics": {
                "random_forest_accuracy": round(accuracy_rf, 4),
                "gradient_boosting_accuracy": round(accuracy_gb, 4),
                "features_used": len(feature_cols),
                "test_samples": len(y_test),
                "rf_confusion_matrix": cm_rf.tolist(),
                "gb_confusion_matrix": cm_gb.tolist(),
                "top_features": [{"feature": feat, "importance": round(imp, 4)} 
                               for feat, imp in top_features]
            },
            "plot": plot_data
        }
        
    except Exception as e:
        return {
            "experiment": "Ensemble Methods (Bagging vs Boosting)",
            "error": str(e),
            "plot": ""
        }
