# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')  # Non-GUI backend for Flask/FastAPI
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# import base64
# import io
# import os
# from typing import Optional


# def _fig_to_base64(fig) -> str:
#     """Convert matplotlib figure to base64 string."""
#     buffer = io.BytesIO()
#     fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
#     buffer.seek(0)
#     data = base64.b64encode(buffer.getvalue()).decode()
#     plt.close(fig)
#     return data


# def _add_caption(fig, caption: str):
#     """Add a caption/insight text below a matplotlib figure."""
#     fig.text(
#         0.5, -0.05, caption,
#         ha='center', va='top',
#         fontsize=10, color='dimgray',
#         wrap=True,
#         bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke", edgecolor="lightgray")
#     )


# def run(age: Optional[int] = None):
#     """
#     Generate visualizations and dataset insights for Heart Disease Prediction.
#     Each plot includes its own descriptive caption inside the image.
#     """
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
#     df = pd.read_csv(data_path)

#     # ---- Replace numeric codes with human-readable labels ----
#     mappings = {
#         "Sex": {0: "Female", 1: "Male"},
#         "Chest pain type": {
#             1: "Typical angina", 2: "Atypical angina",
#             3: "Non-anginal pain", 4: "Asymptomatic"
#         },
#         "FBS over 120": {0: "False", 1: "True"},
#         "Exercise angina": {0: "No", 1: "Yes"},
#         "EKG results": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
#         "Slope of ST": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
#         "Thallium": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}
#     }

#     for col, mapping in mappings.items():
#         if col in df.columns:
#             df[col] = df[col].map(mapping).fillna(df[col])

#     stats = {
#         "shape": df.shape,
#         "columns": df.columns.tolist(),
#         "missing_values": df.isnull().sum().to_dict(),
#         "data_types": df.dtypes.astype(str).to_dict()
#     }

#     filtered_by = None
#     if age is not None and "Age" in df.columns:
#         df_filtered = df[df["Age"] == age]
#         filtered_by = {"Age": age}
#         if df_filtered.empty:
#             return {
#                 "experiment": "Heart Disease Data Visualization",
#                 "metrics": stats,
#                 "filtered_by": filtered_by,
#                 "plots": {},
#                 "message": f"No records found for Age={age}"
#             }
#     else:
#         df_filtered = df.copy()

#     plots, descriptions = {}, {}

#     # 1️⃣ Pie Chart: Chest Pain Type
#     if "Chest pain type" in df_filtered.columns:
#         fig, ax = plt.subplots(figsize=(5, 5))
#         counts = df_filtered["Chest pain type"].value_counts()
#         ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
#         ax.set_title("Distribution of Chest Pain Types", fontsize=13)
#         caption = "Shows proportions of different chest pain types. Typical and atypical angina often indicate heart-related issues."
#         _add_caption(fig, caption)
#         plots["chest_pain_pie"] = _fig_to_base64(fig)
#         descriptions["chest_pain_pie"] = caption

#     # 2️⃣ Bar Chart: Gender Distribution
#     if "Sex" in df_filtered.columns:
#         fig, ax = plt.subplots(figsize=(6, 4))
#         sns.countplot(data=df_filtered, x="Sex", palette="Set2", ax=ax)
#         ax.set_title("Gender Distribution of Patients", fontsize=13)
#         ax.set_xlabel("Gender")
#         ax.set_ylabel("Count")
#         caption = "Compares the number of male and female patients in the dataset."
#         _add_caption(fig, caption)
#         plots["gender_bar"] = _fig_to_base64(fig)
#         descriptions["gender_bar"] = caption

#     # 3️⃣ Line Chart: Cholesterol by Age
#     if "Age" in df_filtered.columns and "Cholesterol" in df_filtered.columns:
#         avg_chol = df_filtered.groupby("Age")["Cholesterol"].mean()
#         fig, ax = plt.subplots(figsize=(7, 4))
#         ax.plot(avg_chol.index, avg_chol.values, color="teal", marker="o")
#         ax.set_title("Average Cholesterol by Age", fontsize=13)
#         ax.set_xlabel("Age")
#         ax.set_ylabel("Mean Cholesterol Level")
#         caption = "Shows how average cholesterol levels vary across different age groups."
#         _add_caption(fig, caption)
#         plots["cholesterol_line"] = _fig_to_base64(fig)
#         descriptions["cholesterol_line"] = caption

#     # 4️⃣ Bar Chart: Exercise Angina vs Heart Disease
#     if "Exercise angina" in df_filtered.columns and "Heart Disease" in df_filtered.columns:
#         fig, ax = plt.subplots(figsize=(6, 4))
#         sns.countplot(data=df_filtered, x="Exercise angina", hue="Heart Disease", palette="coolwarm", ax=ax)
#         ax.set_title("Exercise Angina vs Heart Disease", fontsize=13)
#         ax.set_xlabel("Exercise-Induced Angina")
#         ax.set_ylabel("Patient Count")
#         caption = "Patients with exercise-induced angina are more likely to have heart disease, as shown by the taller 'Presence' bars under 'Yes'."
#         _add_caption(fig, caption)
#         plots["angina_bar"] = _fig_to_base64(fig)
#         descriptions["angina_bar"] = caption

#     # 5️⃣ Correlation Heatmap
#     try:
#         df_encoded = df_filtered.copy()
#         for col in df_encoded.columns:
#             if df_encoded[col].dtype == "object":
#                 df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

#         corr = df_encoded.corr().round(2)
#         plt.figure(figsize=(9, 6))
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
#         plt.title("Correlation Heatmap", fontsize=14)
#         fig = plt.gcf()
#         caption = "Displays correlation between features. Positive = increase together; negative = inverse relationship."
#         _add_caption(fig, caption)
#         plots["correlation_heatmap"] = _fig_to_base64(fig)
#         descriptions["correlation_heatmap"] = caption
#     except Exception:
#         pass

#     return {
#         "experiment": "Heart Disease Data Visualization",
#         "metrics": stats,
#         "filtered_by": filtered_by,
#         "plots": plots,
#         "descriptions": descriptions
#     }


# # ---- CLI usage for local testing ----
# if __name__ == "__main__":
#     try:
#         user = input("Enter age to filter by (press Enter to skip): ").strip()
#     except Exception:
#         user = ""
#     age_val = int(user) if user.isdigit() else None
#     result = run(age=age_val)

#     out_dir = os.path.join(os.path.dirname(__file__), "..", "static")
#     os.makedirs(out_dir, exist_ok=True)

#     plots = result.get("plots", {})
#     if not plots:
#         print("No plots were generated. Message:", result.get("message", ""))
#     else:
#         for name, b64 in plots.items():
#             filename = os.path.join(out_dir, f"{name}" + (f"_age_{age_val}" if age_val is not None else "") + ".png")
#             with open(filename, "wb") as f:
#                 f.write(base64.b64decode(b64))
#             print(f"✅ Saved: {filename}")


import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Flask/FastAPI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import base64
import io
import os
from typing import Optional


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return data


def _add_caption(fig, caption: str):
    """Add a caption/insight text below a matplotlib figure."""
    fig.text(
        0.5, -0.05, caption,
        ha='center', va='top',
        fontsize=10, color='dimgray',
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke", edgecolor="lightgray")
    )


def run(age: Optional[int] = None):
    """
    Generate visualizations and dataset insights for Heart Disease Prediction.
    Each plot includes its own descriptive caption inside the image.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
    df = pd.read_csv(data_path)

    # ---- Replace numeric codes with human-readable labels ----
    mappings = {
        "Sex": {0: "Female", 1: "Male"},
        "Chest pain type": {
            1: "Typical angina", 2: "Atypical angina",
            3: "Non-anginal pain", 4: "Asymptomatic"
        },
        "FBS over 120": {0: "False", 1: "True"},
        "Exercise angina": {0: "No", 1: "Yes"},
        "EKG results": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
        "Slope of ST": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
        "Thallium": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])

    # ---- Dataset summary ----
    stats = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }

    # ---- Filter by age if provided ----
    filtered_by = None
    if age is not None and "Age" in df.columns:
        df_filtered = df[df["Age"] == age]
        filtered_by = {"Age": age}
        if df_filtered.empty:
            return {
                "experiment": "Heart Disease Data Visualization",
                "metrics": stats,
                "filtered_by": filtered_by,
                "plots": {},
                "message": f"No records found for Age={age}"
            }
    else:
        df_filtered = df.copy()

    plots, descriptions = {}, {}

    # 1️⃣ Pie Chart: Chest Pain Type
    if "Chest pain type" in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        counts = df_filtered["Chest pain type"].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax.set_title("Distribution of Chest Pain Types", fontsize=13)
        caption = "Shows proportions of different chest pain types. Typical and atypical angina often indicate heart-related issues."
        _add_caption(fig, caption)
        plots["chest_pain_pie"] = _fig_to_base64(fig)
        descriptions["Chest Pain Distribution"] = caption

    # 2️⃣ Bar Chart: Gender Distribution
    if "Sex" in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_filtered, x="Sex", palette="Set2", ax=ax)
        ax.set_title("Gender Distribution of Patients", fontsize=13)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        caption = "Compares the number of male and female patients in the dataset."
        _add_caption(fig, caption)
        plots["gender_bar"] = _fig_to_base64(fig)
        descriptions["Gender Distribution"] = caption

    # 3️⃣ Line Chart: Cholesterol by Age
    if "Age" in df_filtered.columns and "Cholesterol" in df_filtered.columns:
        avg_chol = df_filtered.groupby("Age")["Cholesterol"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(avg_chol.index, avg_chol.values, color="teal", marker="o")
        ax.set_title("Average Cholesterol by Age", fontsize=13)
        ax.set_xlabel("Age")
        ax.set_ylabel("Mean Cholesterol Level")
        caption = "Shows how average cholesterol levels vary across different age groups."
        _add_caption(fig, caption)
        plots["cholesterol_line"] = _fig_to_base64(fig)
        descriptions["Cholesterol vs Age"] = caption

    # 4️⃣ Bar Chart: Exercise Angina vs Heart Disease
    if "Exercise angina" in df_filtered.columns and "Heart Disease" in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_filtered, x="Exercise angina", hue="Heart Disease", palette="coolwarm", ax=ax)
        ax.set_title("Exercise Angina vs Heart Disease", fontsize=13)
        ax.set_xlabel("Exercise-Induced Angina")
        ax.set_ylabel("Patient Count")
        caption = "Patients with exercise-induced angina are more likely to have heart disease, as shown by the taller 'Presence' bars under 'Yes'."
        _add_caption(fig, caption)
        plots["angina_bar"] = _fig_to_base64(fig)
        descriptions["Exercise Angina vs Heart Disease"] = caption

    # 5️⃣ Correlation Heatmap
    try:
        df_encoded = df_filtered.copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == "object":
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        corr = df_encoded.corr().round(2)
        plt.figure(figsize=(9, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
        plt.title("Correlation Heatmap", fontsize=14)
        fig = plt.gcf()
        caption = "Displays correlation between features. Positive = increase together; negative = inverse relationship."
        _add_caption(fig, caption)
        plots["correlation_heatmap"] = _fig_to_base64(fig)
        descriptions["Correlation Heatmap"] = caption
    except Exception:
        pass

    return {
        "experiment": "Heart Disease Data Visualization",
        "metrics": stats,
        "filtered_by": filtered_by,
        "plots": plots,
        "descriptions": descriptions
    }


# ---- CLI usage for local testing ----
if __name__ == "__main__":
    try:
        user = input("Enter age to filter by (press Enter to skip): ").strip()
    except Exception:
        user = ""
    age_val = int(user) if user.isdigit() else None
    result = run(age=age_val)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    os.makedirs(out_dir, exist_ok=True)

    print("\n📊 Experiment:", result.get("experiment", "N/A"))
    print("📁 Dataset Shape:", result["metrics"]["shape"])
    if result.get("filtered_by"):
        print("🔍 Filter Applied:", result["filtered_by"])
    print("\n🧾 Dataset Columns:", ", ".join(result["metrics"]["columns"]))

    plots = result.get("plots", {})
    if not plots:
        print("\n❌ No plots generated. Message:", result.get("message", ""))
    else:
        print("\n✅ Generated Visualizations and Insights:")
        for name, caption in result["descriptions"].items():
            filename = os.path.join(out_dir, f"{name.replace(' ', '_').lower()}" + (f"_age_{age_val}" if age_val is not None else "") + ".png")
            with open(filename, "wb") as f:
                f.write(base64.b64decode(result["plots"][name.replace(' ', '_').lower()]))
            print(f"\n📈 {name}")
            print(f"   ➤ Insight: {caption}")
            print(f"   💾 Saved to: {filename}")
