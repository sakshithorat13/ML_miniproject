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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import numpy as np

def _fig_to_base64(fig):
    """Convert matplotlib figure to base64."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return data


def run():
    try:
        # Load dataset
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
        df = pd.read_csv(data_path)

        target_col = 'Heart Disease'
        feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols].copy()
        y = df[target_col]

        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        X = X.fillna(X.mean())

        # Encode target
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Scale features
        X_scaled = StandardScaler().fit_transform(X)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Random Forest (Bagging) ---
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
        acc_rf = accuracy_score(y_test, y_pred_rf)

        # --- Gradient Boosting (Boosting) ---
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
        acc_gb = accuracy_score(y_test, y_pred_gb)

        # --- Plot: Probability Distribution ---
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(y_prob_rf, fill=True, color='orange', label='Random Forest')
        sns.kdeplot(y_prob_gb, fill=True, color='teal', label='Gradient Boosting')

        plt.axvline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
        plt.title("Prediction Probability Distribution for Heart Disease", fontsize=12)
        plt.xlabel("Predicted Probability (Class = 1 - Heart Disease)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plot_data = _fig_to_base64(fig)

        # --- Top Features from Random Forest ---
        feature_importance = rf_model.feature_importances_
        top_features = sorted(zip(feature_cols, feature_importance), key=lambda x: x[1], reverse=True)[:5]

        # --- Insights Section ---
        insights_html = f"""
        <h4>💡 Insights and Interpretation</h4>
        <p>The ensemble comparison shows distinct confidence patterns between <b>Random Forest</b> and <b>Gradient Boosting</b>.</p>
        <p><b>Random Forest</b> demonstrates a broader, smoother distribution—indicating balanced predictions with moderate confidence across cases.</p>
        <p><b>Gradient Boosting</b> displays sharper peaks, meaning it makes more confident predictions and focuses on harder-to-classify samples.</p>
        <p>The vertical red dashed line marks the <b>decision threshold (0.5)</b>, separating predicted healthy vs. heart disease cases.</p>
        <p>Both models achieved comparable accuracy: <b>Random Forest = {acc_rf * 100:.2f}%</b> and <b>Gradient Boosting = {acc_gb * 100:.2f}%</b>.</p>
        <p>The top contributing features were <b>{top_features[0][0]}</b>, <b>{top_features[1][0]}</b>, and <b>{top_features[2][0]}</b>, 
        showing strong influence on heart disease prediction.</p>
        <p>In practice, this comparison helps visualize model confidence and uncertainty—critical for healthcare decision support.</p>
        """

        overview_html = """
        <h4>🧠 Ensemble Method Overview</h4>
        <p><b>Random Forest</b> uses <b>Bagging</b> to combine multiple decision trees trained independently, reducing variance.</p>
        <p><b>Gradient Boosting</b> applies <b>Boosting</b> by training trees sequentially, where each corrects the mistakes of the previous ones.</p>
        <p>This complementary behavior makes ensemble methods powerful, stable, and interpretable for clinical applications.</p>
        """

        return {
            "experiment": "Ensemble Learning — Probability Distribution (Bagging vs Boosting)",
            "metrics": {
                "Random Forest Accuracy (%)": round(acc_rf * 100, 2),
                "Gradient Boosting Accuracy (%)": round(acc_gb * 100, 2),
                "Top Features": [f"{f[0]} ({f[1]:.3f})" for f in top_features]
            },
            "plots": {
                "Probability Distribution": plot_data
            },
            "insights_table": [
                {"Graph": "💬 Insights and Interpretation", "Insight": insights_html},
                {"Graph": "🧠 Ensemble Overview", "Insight": overview_html}
            ]
        }

    except Exception as e:
        return {
            "experiment": "Ensemble Learning — Probability Distribution (Bagging vs Boosting)",
            "error": str(e),
            "plots": {}
        }
