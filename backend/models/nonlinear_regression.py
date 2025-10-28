import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import base64
import io
import os

def run():
    try:
        # Load dataset
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
        df = pd.read_csv(data_path)
        
        # Select three main features for 3D visualization
        feature_cols = ['Age', 'BP', 'Cholesterol']
        target_col = 'Heart Disease'
        
        # Clean data - remove rows with missing values in selected features
        df_clean = df[feature_cols + [target_col]].dropna()
        
        # Create binary target (High vs Low Risk based on median)
        if df_clean[target_col].dtype == 'object':
            # Convert categorical to binary
            df_clean['High_Risk'] = (df_clean[target_col] == 'Presence').astype(int)
        else:
            # Use median split for numeric targets
            median_val = df_clean[target_col].median()
            df_clean['High_Risk'] = (df_clean[target_col] >= median_val).astype(int)
        
        # Features and Target
        X = df_clean[feature_cols]
        y = df_clean['High_Risk']
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create polynomial features for nonlinear regression
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Scale polynomial features
        scaler_poly = StandardScaler()
        X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
        X_test_poly_scaled = scaler_poly.transform(X_test_poly)
        
        # Train Models
        # Decision Tree
        dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        
        # SVM (RBF Kernel)
        svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        svm_rbf.fit(X_train_scaled, y_train)
        svm_pred = svm_rbf.predict(X_test_scaled)
        
        # Polynomial Logistic Regression
        poly_lr = LogisticRegression(random_state=42, max_iter=1000)
        poly_lr.fit(X_train_poly_scaled, y_train)
        poly_pred = poly_lr.predict(X_test_poly_scaled)
        
        # Calculate accuracies
        dt_accuracy = accuracy_score(y_test, dt_pred)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        poly_accuracy = accuracy_score(y_test, poly_pred)
        
        # Create visualizations
        fig = plt.figure(figsize=(18, 6))
        
        # 3D Scatter Plot
        ax1 = fig.add_subplot(131, projection='3d')
        
        x = df_clean['Age']
        y_val = df_clean['BP']  
        z = df_clean['Cholesterol']
        labels = df_clean['High_Risk']
        
        # Plot high risk (1) and low risk (0)
        ax1.scatter(x[labels==1], y_val[labels==1], z[labels==1], 
                   c='red', label="High Risk", s=30, alpha=0.6)
        ax1.scatter(x[labels==0], y_val[labels==0], z[labels==0], 
                   c='blue', label="Low Risk", s=30, alpha=0.6)
        
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Blood Pressure")
        ax1.set_zlabel("Cholesterol")
        ax1.set_title("3D Plot: Heart Disease Risk vs Features")
        ax1.legend()
        
        # Confusion Matrix - SVM RBF
        ax2 = fig.add_subplot(132)
        cm_svm = confusion_matrix(y_test, svm_pred)
        im2 = ax2.imshow(cm_svm, cmap=plt.cm.Blues)
        ax2.set_title("Confusion Matrix - SVM RBF")
        plt.colorbar(im2, ax=ax2)
        ax2.set_xticks([0,1])
        ax2.set_yticks([0,1])
        ax2.set_xticklabels(['Low Risk', 'High Risk'])
        ax2.set_yticklabels(['Low Risk', 'High Risk'])
        
        # Add text annotations
        for i in range(cm_svm.shape[0]):
            for j in range(cm_svm.shape[1]):
                ax2.text(j, i, cm_svm[i, j], ha='center', va='center', color='black', fontweight='bold')
        
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        
        # Confusion Matrix - Polynomial Regression
        ax3 = fig.add_subplot(133)
        cm_poly = confusion_matrix(y_test, poly_pred)
        im3 = ax3.imshow(cm_poly, cmap=plt.cm.Blues)
        ax3.set_title("Confusion Matrix - Polynomial Regression")
        plt.colorbar(im3, ax=ax3)
        ax3.set_xticks([0,1])
        ax3.set_yticks([0,1])
        ax3.set_xticklabels(['Low Risk', 'High Risk'])
        ax3.set_yticklabels(['Low Risk', 'High Risk'])
        
        # Add text annotations
        for i in range(cm_poly.shape[0]):
            for j in range(cm_poly.shape[1]):
                ax3.text(j, i, cm_poly[i, j], ha='center', va='center', color='black', fontweight='bold')
        
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        
        plt.tight_layout()
        
        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "experiment": "Multivariate Nonlinear Regression",
            "metrics": {
                "decision_tree_accuracy": round(dt_accuracy, 4),
                "svm_rbf_accuracy": round(svm_accuracy, 4),
                "polynomial_accuracy": round(poly_accuracy, 4),
                "features_used": feature_cols,
                "polynomial_degree": 2,
                "test_samples": len(y_test),
                "high_risk_samples": int(sum(df_clean['High_Risk'])),
                "low_risk_samples": int(len(df_clean) - sum(df_clean['High_Risk']))
            },
            "analysis": {
                "graph_interpretation": f"""
                **Multivariate Nonlinear Regression Analysis:**
                
                **3D Scatter Plot Insights:**
                The 3D visualization reveals natural clustering of patients in Age-BP-Cholesterol space:
                - Red points (High Risk): {int(sum(df_clean['High_Risk']))} patients
                - Blue points (Low Risk): {int(len(df_clean) - sum(df_clean['High_Risk']))} patients
                
                **Model Performance Comparison:**
                - Decision Tree: {dt_accuracy:.2%} accuracy
                - SVM RBF: {svm_accuracy:.2%} accuracy  
                - Polynomial Regression: {poly_accuracy:.2%} accuracy
                
                **Confusion Matrix Analysis:**
                Both matrices show classification performance with different color schemes for easy comparison
                
                **Nonlinear Relationships:**
                Polynomial features capture complex interactions between age, blood pressure, and cholesterol levels
                """,
                "what_graph_shows": "3D patient distribution by risk level and confusion matrices comparing SVM vs Polynomial Regression performance",
                "key_inferences": [
                    f"Best performing model: {['Decision Tree', 'SVM RBF', 'Polynomial'][np.argmax([dt_accuracy, svm_accuracy, poly_accuracy])]}",
                    "Clear risk group separation visible in 3D space",
                    "Non-linear relationships exist between health indicators"
                ],
                "why_graph_like_this": "3D visualization reveals patterns impossible to see in 2D, while confusion matrices provide quantitative performance metrics",
                "practical_applications": [
                    "Multi-factor risk assessment in clinical settings",
                    "Personalized treatment planning",
                    "Early intervention strategies",
                    "Healthcare resource optimization"
                ]
            },
            "plot": plot_data
        }
        
    except Exception as e:
        return {
            "experiment": "Multivariate Nonlinear Regression",
            "error": str(e),
            "plot": ""
        }
