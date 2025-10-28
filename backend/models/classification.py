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
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os


def run():
    # --- Load dataset ---
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    df = pd.read_csv(data_path)

    # Identify target and features
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].fillna(df[feature_cols].mean())
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

    # --- Confusion Matrix Plot (smaller image) ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
    plt.title('Confusion Matrix - Decision Tree Classifier', fontsize=13)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    buffer_cm = io.BytesIO()
    plt.savefig(buffer_cm, format='png', dpi=120)
    buffer_cm.seek(0)
    cm_plot_data = base64.b64encode(buffer_cm.getvalue()).decode()
    plt.close()

    # --- Decision Tree Structure Plot ---
    plt.figure(figsize=(14, 9))
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=[str(c) for c in sorted(y.unique())],
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title('Decision Tree Model Visualization', fontsize=14)
    plt.tight_layout()

    buffer_tree = io.BytesIO()
    plt.savefig(buffer_tree, format='png', dpi=120)
    buffer_tree.seek(0)
    tree_plot_data = base64.b64encode(buffer_tree.getvalue()).decode()
    plt.close()

    # --- Feature Importance Plot ---
    feature_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette="crest")
    plt.title('Feature Importance (Gini-based)', fontsize=13)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()

    buffer_imp = io.BytesIO()
    plt.savefig(buffer_imp, format='png', dpi=120)
    buffer_imp.seek(0)
    imp_plot_data = base64.b64encode(buffer_imp.getvalue()).decode()
    plt.close()

    # --- Extract Top 3 Important Features ---
    top_features = feature_importances.head(3)
    top_feats_str = ", ".join([f"<b>{feat}</b>" for feat in top_features.index])

    # --- Insights & Interpretation ---
    insight_text = """
    
    <h4>🌳 Decision Tree Overview:</h4>
    <p>
    A <b>Decision Tree Classifier</b> is a supervised machine learning algorithm used to make predictions 
    based on feature conditions. It works by splitting the dataset into branches according to feature thresholds 
    that best separate the target classes — in this case, <b>presence or absence of heart disease</b>.
    </p>
    <p>
    Each node in the tree represents a decision rule based on patient attributes such as <b>blood pressure, cholesterol, 
    chest pain type, and age</b>. The model selects these rules to maximize classification accuracy using measures like 
    <b>information gain</b> or <b>Gini impurity</b>.
    </p>

    <h4>📊 Confusion Matrix Interpretation:</h4>
    <p>
    The <b>Confusion Matrix</b> provides a summary of prediction results. The diagonal values indicate correct predictions 
    for both positive (heart disease present) and negative (no heart disease) cases. Off-diagonal values show misclassifications.
    </p>
    <ul>
      <li><b>True Positives (TP):</b> Patients correctly predicted with heart disease.</li>
      <li><b>True Negatives (TN):</b> Patients correctly predicted as healthy.</li>
      <li><b>False Positives (FP):</b> Healthy patients incorrectly labeled as heart disease cases.</li>
      <li><b>False Negatives (FN):</b> Heart disease patients incorrectly predicted as healthy.</li>
    </ul>

    <h4>💡 Insights and Interpretation:</h4>
    <p>-The Decision Tree model achieved an accuracy of <b>70.37%</b>, 
                indicating fair predictive strength on this dataset.<br>
               - <b>Chest Pain Type (cp)</b> showed the highest information gain, 
                making it the primary factor in predicting heart disease.<br>
                -<b>Thallium Test Result (thal)</b> and <b>ST Depression (oldpeak)</b> 
                were the next most influential features, reflecting heart stress and blood flow abnormalities.<br>
               - The tree structure clearly shows how decisions are made step-by-step, 
                helping to identify combinations of medical factors linked to heart disease.<br>
                -The confusion matrix shows that while most predictions are correct, 
                some overlap between healthy and heart disease cases still occurs.<br>
                -Overall, the model provides an interpretable, rule-based approach 
                that aligns well with clinical reasoning.
                </p>
    
    """.format(accuracy * 100)

    # --- Return Output ---
    return {
        "experiment": "Decision Tree Classification",
        "metrics": {
            "accuracy": round(accuracy, 4),
            "target_variable": target_col,
            "features_used": len(feature_cols)
        },
        "plots": {
            "confusion_matrix": cm_plot_data,
            "decision_tree": tree_plot_data
        },
        "insight": {
            "Graph": "💬 Decision Tree Classification Insights",
            "Insight": insight_text
        }
    }
