import streamlit as st
import tempfile
import os
import joblib
import lizard
import pandas as pd
from radon.metrics import h_visit
from radon.raw import analyze

# =====================
# FEATURE CONFIG
# =====================
FEATURE_COLUMNS = [
    "LOC_EXECUTABLE",
    "LOC_TOTAL",
    "CYCLOMATIC_COMPLEXITY",
    "HALSTEAD_VOLUME",
    "HALSTEAD_DIFFICULTY",
    "NUM_OPERATORS",
    "NUM_OPERANDS"
]

def extract_metrics(path):
    data = []

    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(('.py', '.java', '.c', '.cpp'))
    ]

    for file_path in files:
        with open(file_path, "r", errors="ignore") as f:
            code = f.read()

        # RADON
        raw = analyze(code)
        halstead = h_visit(code)

        total_loc = raw.loc

        h_volume = halstead.total.volume
        h_difficulty = halstead.total.difficulty
        num_operators = halstead.total.N1
        num_operands = halstead.total.N2

        # LIZARD
        for file_info in lizard.analyze_files([file_path]):
            for func in file_info.function_list:
                data.append({
                    "function": func.name,
                    "file": file_info.filename,
                    "LOC_EXECUTABLE": func.nloc,
                    "LOC_TOTAL": total_loc,
                    "CYCLOMATIC_COMPLEXITY": func.cyclomatic_complexity,
                    "HALSTEAD_VOLUME": h_volume,
                    "HALSTEAD_DIFFICULTY": h_difficulty,
                    "NUM_OPERATORS": num_operators,
                    "NUM_OPERANDS": num_operands
                })

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("Tidak ada function terdeteksi")

    return df

st.title("Software Defect Prediction (7 Features)")

uploaded_file = st.file_uploader(
    "Upload source code (.py, .java, .c, .cpp)",
    type=["py", "java", "c", "cpp"]
)

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("File uploaded")

        df_metrics = extract_metrics(tmp)
        st.subheader("Extracted Metrics")
        st.dataframe(df_metrics)

        model = joblib.load("model/rf_7_features.pkl")
        scaler = joblib.load("model/scaler_7_features.pkl")

        X = df_metrics[FEATURE_COLUMNS]
        X_scaled = scaler.transform(X)

        THRESHOLD = 0.3
        proba = model.predict_proba(X_scaled)[:, 1]

        df_metrics["Defect_Probability"] = proba
        df_metrics["Defect_Prediction"] = (proba >= THRESHOLD).astype(int)

        st.subheader("Prediction Result")
        st.dataframe(df_metrics)
