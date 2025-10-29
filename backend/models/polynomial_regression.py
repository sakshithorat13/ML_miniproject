# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64
# import io
# import os

# def run():
#     try:
#         # Load the dataset
#         data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#         df = pd.read_csv(data_path)
        
#         # Data Preprocessing
#         target_col = 'Heart Disease'
#         feature_cols = [col for col in df.columns if col != target_col]
        
#         # Separate features (X) and target (y)
#         X = df[feature_cols].copy()
#         y = df[target_col]
        
#         # Handle categorical features
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))
        
#         # Fill missing values
#         X = X.fillna(X.mean())
        
#         # Convert target to binary if categorical
#         if y.dtype == 'object':
#             le_target = LabelEncoder()
#             y_binary = le_target.fit_transform(y)
#         else:
#             y_binary = y
        
#         # Scale features
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Split data into training and testing sets (80% train, 20% test)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
#         )
        
#         # Implement Bagging (Random Forest)
#         print("Training Random Forest Model...")
#         rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#         rf_model.fit(X_train, y_train)
#         y_pred_rf = rf_model.predict(X_test)
        
#         # Implement Boosting (Gradient Boosting)
#         print("Training Gradient Boosting Model...")
#         gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
#         gb_model.fit(X_train, y_train)
#         y_pred_gb = gb_model.predict(X_test)
        
#         # Calculate accuracies
#         accuracy_rf = accuracy_score(y_test, y_pred_rf)
#         accuracy_gb = accuracy_score(y_test, y_pred_gb)
        
#         # Create confusion matrices
#         cm_rf = confusion_matrix(y_test, y_pred_rf)
#         cm_gb = confusion_matrix(y_test, y_pred_gb)
        
#         # Create visualizations
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
#         # Random Forest Confusion Matrix
#         sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
#                    xticklabels=['Not Churned', 'Churned'], 
#                    yticklabels=['Not Churned', 'Churned'], ax=ax1)
#         ax1.set_title('Random Forest Confusion Matrix')
#         ax1.set_ylabel('Actual')
#         ax1.set_xlabel('Predicted')
        
#         # Gradient Boosting Confusion Matrix
#         sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens',
#                    xticklabels=['Not Churned', 'Churned'], 
#                    yticklabels=['Not Churned', 'Churned'], ax=ax2)
#         ax2.set_title('Gradient Boosting Confusion Matrix')
#         ax2.set_ylabel('Actual')
#         ax2.set_xlabel('Predicted')
        
#         plt.tight_layout()
        
#         # Save plot to base64
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
#         buffer.seek(0)
#         plot_data = base64.b64encode(buffer.getvalue()).decode()
#         plt.close()
        
#         # Get feature importance from Random Forest
#         feature_importance = rf_model.feature_importances_
#         top_features = sorted(zip(feature_cols, feature_importance), 
#                             key=lambda x: x[1], reverse=True)[:5]
        
#         return {
#             "experiment": "Ensemble Methods (Bagging vs Boosting)",
#             "metrics": {
#                 "random_forest_accuracy": round(accuracy_rf, 4),
#                 "gradient_boosting_accuracy": round(accuracy_gb, 4),
#                 "features_used": len(feature_cols),
#                 "test_samples": len(y_test),
#                 "rf_confusion_matrix": cm_rf.tolist(),
#                 "gb_confusion_matrix": cm_gb.tolist(),
#                 "top_features": [{"feature": feat, "importance": round(imp, 4)} 
#                                for feat, imp in top_features]
#             },
#             "analysis": {
#                 "graph_interpretation": f"""
#                 **Confusion Matrix Analysis:**
                
#                 The side-by-side confusion matrices reveal important insights about ensemble methods:
                
#                 **Random Forest (Bagging) Performance:**
#                 - True Negatives: {cm_rf[0,0]} patients correctly identified as low risk
#                 - False Positives: {cm_rf[0,1]} patients incorrectly flagged as high risk
#                 - False Negatives: {cm_rf[1,0]} high-risk patients missed (critical misses)
#                 - True Positives: {cm_rf[1,1]} high-risk patients correctly identified
#                 - Accuracy: {accuracy_rf:.2%}
                
#                 **Gradient Boosting Performance:**
#                 - True Negatives: {cm_gb[0,0]} patients correctly identified as low risk
#                 - False Positives: {cm_gb[0,1]} patients incorrectly flagged as high risk  
#                 - False Negatives: {cm_gb[1,0]} high-risk patients missed
#                 - True Positives: {cm_gb[1,1]} high-risk patients correctly identified
#                 - Accuracy: {accuracy_gb:.2%}
                
#                 **Key Insights:**
                
#                 1. **Algorithm Comparison:** {'Gradient Boosting' if accuracy_gb > accuracy_rf else 'Random Forest'} shows superior performance with {max(accuracy_gb, accuracy_rf):.2%} accuracy
                
#                 2. **Clinical Significance:** False negatives are particularly concerning as they represent high-risk patients who might not receive necessary treatment
                
#                 3. **Model Behavior:**
#                    - Random Forest uses parallel decision trees (bagging) for robust predictions
#                    - Gradient Boosting builds sequential trees, learning from previous errors
#                    - Both methods help reduce overfitting compared to single decision trees
                
#                 4. **Feature Importance:** The top predictive features are {', '.join([f[0] for f in top_features[:3]])}, indicating these are most crucial for heart disease prediction
                
#                 **Predictions & Recommendations:**
#                 - Use the {'gradient boosting' if accuracy_gb > accuracy_rf else 'random forest'} model for deployment
#                 - Consider ensemble voting between both models for critical decisions
#                 - Focus on improving recall for high-risk patients to minimize false negatives
#                 - Monitor model performance with additional validation data
#                 """,
#                 "what_graph_shows": "Two confusion matrices comparing Random Forest vs Gradient Boosting classification performance on heart disease prediction",
#                 "key_inferences": [
#                     f"{'Gradient Boosting' if accuracy_gb > accuracy_rf else 'Random Forest'} achieves better overall accuracy",
#                     f"Both models show {min(accuracy_gb, accuracy_rf):.2%} minimum accuracy threshold",
#                     f"Feature importance ranking: {top_features[0][0]} is most predictive"
#                 ],
#                 "why_graph_like_this": "Confusion matrices use color intensity to show prediction accuracy - darker colors indicate higher values, helping identify where models perform well or struggle",
#                 "practical_applications": [
#                     "Hospital screening systems for early heart disease detection",
#                     "Risk stratification for preventive care programs", 
#                     "Clinical decision support for healthcare providers",
#                     "Insurance risk assessment and premium calculation"
#                 ]
#             },
#             "plot": plot_data
#         }
        
#     except Exception as e:
#         return {
#             "experiment": "Ensemble Methods (Bagging vs Boosting)",
#             "error": str(e),
#             "plot": ""
#         }


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For Flask / non-GUI backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import base64
import io
import os

def run():
    try:
        # 🧹 Prevent overlapping plots from previous executions
        plt.close('all')
        plt.clf()
        plt.rcParams.update({'figure.max_open_warning': 0})

        # ✅ Load dataset
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare_datasetnew.csv')
        df = pd.read_csv(data_path)

        # ✅ Select features and target
        X = df[['Age', 'Treatment Cost', 'Stay Duration']].values
        Y = df['Recovery Days'].values

        # ✅ Handle missing values
        X = np.nan_to_num(X, nan=np.nanmean(X))
        Y = np.nan_to_num(Y, nan=np.nanmean(Y))

        # ✅ Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ✅ Polynomial transformation
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_scaled)

        # ✅ Train regression model
        model = LinearRegression()
        model.fit(X_poly, Y)

        # ✅ Model metrics
        r2_score = model.score(X_poly, Y)
        intercept = round(model.intercept_, 4)
        coeff_count = len(model.coef_)

        # ✅ Create categorical labels for confusion matrix
        median_recovery = np.median(Y)
        y_true = np.where(Y > median_recovery, 1, 0)
        y_pred_continuous = model.predict(X_poly)
        y_pred = np.where(y_pred_continuous > median_recovery, 1, 0)

        # ✅ Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # ✅ Create a new clean figure
        fig = plt.figure(figsize=(12, 5))

        # --- 3D Polynomial Regression Surface ---
        ax1 = fig.add_subplot(121, projection='3d')
        x_surf, y_surf = np.meshgrid(
            np.linspace(X[:, 0].min(), X[:, 0].max(), 50),
            np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
        )

        X_grid_poly = poly.transform(
            scaler.transform(np.c_[
                x_surf.ravel(),
                y_surf.ravel(),
                np.mean(X[:, 2]) * np.ones_like(x_surf.ravel())
            ])
        )

        z_surf = model.predict(X_grid_poly).reshape(x_surf.shape)

        ax1.scatter(X[:, 0], X[:, 1], Y, c='blue', alpha=0.6, edgecolor='k', label='Actual Data')
        ax1.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Treatment Cost')
        ax1.set_zlabel('Recovery Days')
        ax1.set_title('Polynomial Regression Surface')
        ax1.legend()

        # --- Confusion Matrix (Single Plot) ---
        ax2 = fig.add_subplot(122)
        im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax2)
        ax2.set_title('Confusion Matrix - Polynomial Regression')
        classes = ['Low', 'High']
        tick_marks = np.arange(len(classes))
        ax2.set_xticks(tick_marks)
        ax2.set_yticks(tick_marks)
        ax2.set_xticklabels(classes)
        ax2.set_yticklabels(classes)
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')

        # Display numbers inside cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=10)

        plt.tight_layout()

        # ✅ Save to buffer (single image only)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()

        # ✅ Close current figure only
        plt.close(fig)

        # ✅ Return structured response
        return {
            "experiment": "Polynomial Regression (Single Confusion Matrix - Fixed)",
            "metrics": {
                "r2_score": round(r2_score, 4),
                "intercept": intercept,
                "coefficients": coeff_count,
                "features_used": ['Age', 'Treatment Cost', 'Stay Duration'],
                "target": "Recovery Days"
            },
            "analysis": {
                "graph_interpretation": f"""
                **Polynomial Regression (Fixed Version)**

                - The 3D surface (red) represents the nonlinear recovery trend versus Age and Treatment Cost.
                - The confusion matrix shows predicted vs. actual recovery groups (High vs. Low).
                - R² Score: {r2_score:.2f}  
                - Intercept: {intercept}  
                - Total Polynomial Features: {coeff_count}
                """,
                "key_inferences": [
                    "✅ Only ONE confusion matrix is now displayed.",
                    "Polynomial regression effectively captures nonlinear recovery patterns.",
                    "Useful for visualizing complex healthcare relationships."
                ]
            },
            "plot": plot_data
        }

    except Exception as e:
        return {
            "experiment": "Polynomial Regression (Error)",
            "error": str(e),
            "plot": ""
        }
