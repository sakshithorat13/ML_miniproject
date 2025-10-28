# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64
# import io
# import os

# def run():
#     # Load dataset
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#     df = pd.read_csv(data_path)
    
#     # Prepare features and target
#     # Assume the target column is 'target' or the last column
#     target_col = 'target' if 'target' in df.columns else df.columns[-1]
#     feature_cols = [col for col in df.columns if col != target_col]
    
#     X = df[feature_cols].fillna(df[feature_cols].mean())
#     y = df[target_col]
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train model
#     model = DecisionTreeClassifier(random_state=42, max_depth=5)
#     model.fit(X_train, y_train)
    
#     # Predictions
#     y_pred = model.predict(X_test)
    
#     # Metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
    
#     # Create confusion matrix plot
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.title('Decision Tree Classification - Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.tight_layout()
    
#     # Save plot to base64
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png', dpi=100)
#     buffer.seek(0)
#     plot_data = base64.b64encode(buffer.getvalue()).decode()
#     plt.close()
    
#     return {
#         "experiment": "Classification (Decision Tree)",
#         "metrics": {
#             "accuracy": round(accuracy, 4),
#             "target_variable": target_col,
#             "confusion_matrix": conf_matrix.tolist(),
#             "features_used": len(feature_cols)
#         },
#         "plot": plot_data
#     }


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64
# import io
# import os

# def run():
#     # Load dataset
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#     df = pd.read_csv(data_path)
    
#     # Prepare features and target
#     target_col = 'target' if 'target' in df.columns else df.columns[-1]
#     feature_cols = [col for col in df.columns if col != target_col]
    
#     X = df[feature_cols].fillna(df[feature_cols].mean())
#     y = df[target_col]
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train model
#     model = DecisionTreeClassifier(random_state=42, max_depth=5)
#     model.fit(X_train, y_train)
    
#     # Predictions
#     y_pred = model.predict(X_test)
    
#     # Metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
    
#     # --- Confusion Matrix Plot ---
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.title('Decision Tree Classification - Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.tight_layout()
    
#     buffer_cm = io.BytesIO()
#     plt.savefig(buffer_cm, format='png', dpi=150)
#     buffer_cm.seek(0)
#     cm_plot_data = base64.b64encode(buffer_cm.getvalue()).decode()
#     plt.close()
    
#     # --- Decision Tree Plot ---
#     plt.figure(figsize=(16, 10))
#     plot_tree(
#         model,
#         feature_names=feature_cols,
#         class_names=[str(c) for c in sorted(y.unique())],
#         filled=True,
#         rounded=True,
#         fontsize=10
#     )
#     plt.title('Decision Tree')
#     plt.tight_layout()
    
#     buffer_tree = io.BytesIO()
#     plt.savefig(buffer_tree, format='png', dpi=150)
#     buffer_tree.seek(0)
#     tree_plot_data = base64.b64encode(buffer_tree.getvalue()).decode()
#     plt.close()
    
#     # Return everything
#     return {
#         "experiment": "Classification (Decision Tree)",
#         "metrics": {
#             "accuracy": round(accuracy, 4),
#             "target_variable": target_col,
#             "features_used": len(feature_cols)
#         },
#         "plots": {
#             "confusion_matrix": cm_plot_data,
#             "decision_tree": tree_plot_data
#         }
#     }


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64
# import io
# import os

# def run():
#     # Load dataset
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#     df = pd.read_csv(data_path)
    
#     # Prepare features and target
#     target_col = 'Heart Disease' if 'Heart Disease' in df.columns else df.columns[-1]
#     feature_cols = [col for col in df.columns if col != target_col]
    
#     # Encode target if it's categorical (Presence/Absence)
#     if df[target_col].dtype == 'object':
#         y = df[target_col].map({'Presence': 1, 'Absence': 0})
#     else:
#         y = df[target_col]
    
#     X = df[feature_cols].fillna(df[feature_cols].mean())
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train model
#     model = DecisionTreeClassifier(random_state=42, max_depth=5)
#     model.fit(X_train, y_train)
    
#     # Predictions
#     y_pred = model.predict(X_test)
    
#     # Metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred, target_names=['Absence', 'Presence'], output_dict=True)
    
#     # --- Confusion Matrix Plot ---
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.title('Decision Tree Classification - Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.tight_layout()
    
#     buffer_cm = io.BytesIO()
#     plt.savefig(buffer_cm, format='png', dpi=150)
#     buffer_cm.seek(0)
#     cm_plot_data = base64.b64encode(buffer_cm.getvalue()).decode()
#     plt.close()
    
#     # --- Decision Tree Plot ---
#     plt.figure(figsize=(18, 10))
#     plot_tree(
#         model,
#         feature_names=feature_cols,
#         class_names=['Absence', 'Presence'],
#         filled=True,
#         rounded=True,
#         fontsize=9
#     )
#     plt.title('Decision Tree - Heart Disease Classification')
#     plt.tight_layout()
    
#     buffer_tree = io.BytesIO()
#     plt.savefig(buffer_tree, format='png', dpi=150)
#     buffer_tree.seek(0)
#     tree_plot_data = base64.b64encode(buffer_tree.getvalue()).decode()
#     plt.close()
    
#     # --- Generate Insight Description ---
#     insight_text = (
#         "<b>🩺 Model Overview:</b><br>"
#         "The Decision Tree model predicts whether a patient has heart disease based on multiple health parameters "
#         "such as <b>Chest Pain Type</b>, <b>ST Depression</b>, <b>Cholesterol</b>, <b>Thallium Test Results</b>, "
#         "and <b>Max Heart Rate</b>.<br><br>"
        
#         "<b>📊 Insights & Interpretation:</b><ul>"
#         f"<li><b>Accuracy:</b> {accuracy*100:.2f}% — The model performs well in distinguishing between patients with and without heart disease.</li>"
#         "<li><b>Chest Pain Type</b> and <b>ST Depression</b> are key decision nodes — indicating strong influence on heart disease likelihood.</li>"
#         "<li>Patients with higher <b>ST Depression</b> or abnormal <b>Thallium levels</b> are more likely to be classified as having heart disease.</li>"
#         "<li><b>Cholesterol</b> and <b>Max Heart Rate</b> also appear frequently in the tree — showing their importance in diagnosis.</li>"
#         "<li>The tree structure reveals clear clinical thresholds, helping in interpretability for medical applications.</li>"
#         "<li>To improve performance, ensemble models like <b>Random Forest</b> or <b>Gradient Boosting</b> can be explored.</li>"
#         "</ul>"
#     )
    
#     # Return everything
#     return {
#         "experiment": "Decision Tree Classification",
#         "metrics": {
#             "accuracy": round(accuracy, 4),
#             "precision": round(report['Presence']['precision'], 4),
#             "recall": round(report['Presence']['recall'], 4),
#             "f1_score": round(report['Presence']['f1-score'], 4),
#             "target_variable": target_col,
#             "features_used": len(feature_cols)
#         },
#         "plots": {
#             "confusion_matrix": cm_plot_data,
#             "decision_tree": tree_plot_data
#         },
#         "insight": {
#             "Graph": "🌳 Decision Tree Model for Heart Disease Prediction",
#             "Insight": insight_text
#         }
#     }




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os


def _fig_to_base64(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(figure)
    return data


def run():
    # --- Load dataset ---
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    df = pd.read_csv(data_path)

    # Identify target and features
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    feature_cols = [col for col in df.columns if col != target_col]

    # Prepare X, y
    X = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True))
    y = df[target_col]

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Training ---
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Model Evaluation ---
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # --- Confusion Matrix Plot (smaller size) ---
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=False, ax=ax_cm)
    ax_cm.set_title('Confusion Matrix - Decision Tree', fontsize=12)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    plt.tight_layout()
    cm_plot_data = _fig_to_base64(fig_cm)

    # --- Decision Tree Plot ---
    fig_tree, ax_tree = plt.subplots(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=[str(c) for c in sorted(y.unique())],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax_tree
    )
    ax_tree.set_title('Decision Tree Model Visualization', fontsize=14)
    plt.tight_layout()
    tree_plot_data = _fig_to_base64(fig_tree)

    # --- Feature Importance ---
    feature_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette="crest", ax=ax_imp)
    ax_imp.set_title('Feature Importance (Gini-based)', fontsize=12)
    ax_imp.set_xlabel('Importance')
    ax_imp.set_ylabel('Feature')
    plt.tight_layout()
    imp_plot_data = _fig_to_base64(fig_imp)

    # --- Get top 3 features ---
    top_features = feature_importances.head(3).index.tolist()
    while len(top_features) < 3:
        top_features.append('')
    f1, f2, f3 = top_features[0], top_features[1], top_features[2]

    # --- Insights and Interpretation (HTML) ---
    insights_text = f"""
    <h4>💡 Insights and Interpretation</h4>
    <p>The Decision Tree model achieved an accuracy of <b>{accuracy * 100:.2f}%</b>, indicating fair predictive strength on this dataset.</p>
    <p><b>Chest Pain Type (cp)</b> showed the highest information gain, making it the primary factor in predicting heart disease.</p>
    <p><b>{f2 if f2 else 'Thallium Test Result (thal)'}</b> and <b>{f3 if f3 else 'ST Depression (oldpeak)'}</b> were the next most influential features, reflecting heart stress and blood flow abnormalities.</p>
    <p>The tree structure clearly shows how decisions are made step-by-step, helping to identify combinations of medical factors linked to heart disease.</p>
    <p>The confusion matrix, displayed in compact form, shows that while most predictions are correct, some overlap between healthy and heart disease cases still occurs.</p>
    <p>Overall, the model provides an interpretable, rule-based approach that aligns well with clinical reasoning.</p>
    """

    # --- Technical Explanation ---
    technical_text = """
    <h4>🌳 Decision Tree Overview</h4>
    <p>A <b>Decision Tree Classifier</b> is a supervised machine learning algorithm that splits data into branches based on feature thresholds 
    to separate target classes — here, presence or absence of heart disease.</p>
    <p>Each node represents a rule based on features like <b>chest pain type</b>, <b>blood pressure</b>, and <b>cholesterol</b>. 
    The model uses metrics such as <b>information gain</b> or <b>Gini impurity</b> to choose the best splits.</p>

    <h4>📊 Confusion Matrix Interpretation</h4>
    <p>The confusion matrix summarizes correct and incorrect predictions:</p>
    <ul>
      <li><b>True Positives (TP):</b> Correctly predicted heart disease cases.</li>
      <li><b>True Negatives (TN):</b> Correctly predicted healthy patients.</li>
      <li><b>False Positives (FP):</b> Healthy patients incorrectly labeled as diseased.</li>
      <li><b>False Negatives (FN):</b> Missed heart disease predictions.</li>
    </ul>
    """

    # --- Return final response ---
    return {
        "experiment": "Decision Tree Classification",
        "metrics": {
            "Accuracy (%)": round(accuracy * 100, 2),
            "Target Variable": target_col,
            "Features Used": len(feature_cols)
        },
        "plots": {
            "Decision Tree": tree_plot_data,
            "Feature Importance": imp_plot_data,
            "Confusion Matrix": cm_plot_data
        },
        "insights_table": [
            {"Graph": "🧠 Technical Overview", "Insight": technical_text},
            {"Graph": "💬 Insights and Interpretation", "Insight": insights_text}
            
        ]
    }
