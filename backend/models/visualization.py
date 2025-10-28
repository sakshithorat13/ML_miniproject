import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import base64
import io
import os
import numpy as np
from typing import Optional


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return data


def _get_summary_metrics(df: pd.DataFrame) -> dict:
    """Generate clean and summarized dataset metrics."""
    metrics = {}

    # Basic info
    metrics["Total Rows"] = df.shape[0]
    metrics["Total Columns"] = df.shape[1]

    # Data type distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    metrics["Numeric Columns"] = len(numeric_cols)
    metrics["Categorical Columns"] = len(categorical_cols)

    # Missing values
    total_missing = df.isnull().sum().sum()
    metrics["Missing Values"] = int(total_missing)



    # Correlation analysis with target
    target = "Heart Disease"
    if target in df.columns:
        try:
            df_enc = df.copy()
            for col in df_enc.columns:
                if df_enc[col].dtype == "object":
                    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
            corr = df_enc.corr()[target].abs().sort_values(ascending=False)
            top_feature = corr.drop(target, errors='ignore').head(1)
            if not top_feature.empty:
                metrics["Top Correlated Feature"] = f"{top_feature.index[0]} ({top_feature.iloc[0]:.2f})"
            else:
                metrics["Top Correlated Feature"] = "N/A"
        except Exception:
            metrics["Top Correlated Feature"] = "N/A"
    else:
        metrics["Top Correlated Feature"] = "N/A"

    return metrics


def run(age: Optional[int] = None):
    """Generate visualizations and insights for Heart Disease Prediction dataset."""
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

    # --- Improved clean metrics section ---
    stats = _get_summary_metrics(df)

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
                "insights_table": [],
                "message": f"No records found for Age={age}"
            }
    else:
        df_filtered = df.copy()

    plots = {}
    insights_table = []

    # ⿡ Pie Chart: Chest Pain Type
    if "Chest pain type" in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        counts = df_filtered["Chest pain type"].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90,
               colors=sns.color_palette("pastel"))
        ax.set_title("Distribution of Chest Pain Types", fontsize=13)
        plots["chest_pain_pie"] = _fig_to_base64(fig)
        insights_table.append({
            "Graph": "Distribution of Chest Pain Types",
            "Insight": "Shows proportions of different chest pain types. Typical and atypical angina are more likely to indicate heart-related issues."
        })

    # ⿢ Bar Chart: Gender Distribution
    if "Sex" in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.countplot(data=df_filtered, x="Sex", palette="Set2", ax=ax)
        ax.set_title("Gender Distribution of Patients", fontsize=13)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        plots["gender_bar"] = _fig_to_base64(fig)
        insights_table.append({
            "Graph": "Gender Distribution",
            "Insight": "Shows the number of male and female patients. Males are more frequently represented in the dataset."
        })

    # ⿣ Line Chart: Cholesterol by Age
    if "Age" in df_filtered.columns and "Cholesterol" in df_filtered.columns:
        avg_chol = df_filtered.groupby("Age")["Cholesterol"].mean()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(avg_chol.index, avg_chol.values, color="teal", marker="o")
        ax.set_title("Average Cholesterol by Age", fontsize=13)
        ax.set_xlabel("Age")
        ax.set_ylabel("Mean Cholesterol Level")
        plots["cholesterol_line"] = _fig_to_base64(fig)
        insights_table.append({
            "Graph": "Average Cholesterol by Age",
            "Insight": "Displays how average cholesterol levels increase with age, highlighting risk growth in middle-aged and older adults."
        })

    # ⿤ Bar Chart: Exercise Angina vs Heart Disease
    if "Exercise angina" in df_filtered.columns and "Heart Disease" in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.countplot(data=df_filtered, x="Exercise angina", hue="Heart Disease", palette="coolwarm", ax=ax)
        ax.set_title("Exercise Angina vs Heart Disease", fontsize=13)
        ax.set_xlabel("Exercise-Induced Angina")
        ax.set_ylabel("Patient Count")
        plots["angina_bar"] = _fig_to_base64(fig)
        insights_table.append({
            "Graph": "Exercise Angina vs Heart Disease",
            "Insight": "Patients with exercise-induced angina show a higher prevalence of heart disease, visible through taller red bars."
        })

    # ⿥ Histogram: Cholesterol Distribution by Heart Disease
    if "Cholesterol" in df_filtered.columns and "Heart Disease" in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df_filtered, x="Cholesterol", hue="Heart Disease", kde=True,
                     multiple="stack", palette="Set2", ax=ax)
        ax.set_title("Cholesterol Levels vs Heart Disease", fontsize=13)
        ax.set_xlabel("Cholesterol (mg/dl)")
        ax.set_ylabel("Count")
        plots["cholesterol_hist"] = _fig_to_base64(fig)
        insights_table.append({
            "Graph": "Cholesterol Levels vs Heart Disease",
            "Insight": "Compares cholesterol distributions among patients with and without heart disease. Higher cholesterol values are more frequent in patients with heart disease."
        })

    # ⿦ Correlation Heatmap
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
        plots["correlation_heatmap"] = _fig_to_base64(fig)
        insights_table.append({
            "Graph": "Correlation Heatmap",
            "Insight": "Displays correlation between all numeric features. Strong positive/negative relationships indicate predictive potential for heart disease."
        })
    except Exception as e:
        print("Heatmap generation failed:", e)

    return {
        "experiment": "Heart Disease Data Visualization",
        "metrics": stats,
        "filtered_by": filtered_by,
        "plots": plots,
        "insights_table": insights_table
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

    plots = result.get("plots", {})
    if not plots:
        print("No plots were generated. Message:", result.get("message", ""))
    else:
        for name, b64 in plots.items():
            filename = os.path.join(out_dir, f"{name}" + (f"age{age_val}" if age_val is not None else "") + ".png")
            with open(filename, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"✅ Saved: {filename}")

        print("\n📊 Insights Table:")
        for row in result["insights_table"]:
            print(f"- {row['Graph']}: {row['Insight']}")

