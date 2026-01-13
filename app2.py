import streamlit as st
import tempfile
import os
import joblib
import lizard
import pandas as pd
from radon.metrics import h_visit
from radon.raw import analyze
from radon.raw import analyze
from radon.metrics import h_visit
import os 

# FEATURE CONFIG
FEATURE_COLUMNS = [
    "LOC_EXECUTABLE",
    "LOC_TOTAL",
    "CYCLOMATIC_COMPLEXITY",
    "HALSTEAD_VOLUME",
    "HALSTEAD_DIFFICULTY",
    "NUM_OPERATORS",
    "NUM_OPERANDS"
]


def is_python_file(filename):
    return filename.endswith(".py")

def extract_python_metrics(code, filename):
    metrics = {
        "HALSTEAD_VOLUME": None,
        "HALSTEAD_DIFFICULTY": None,
        "HALSTEAD_EFFORT": None
    }

    if not is_python_file(filename):
        return metrics  

    try:
        raw = analyze(code)
        h = h_visit(code)

        if h:
            metrics["HALSTEAD_VOLUME"] = h[0].volume
            metrics["HALSTEAD_DIFFICULTY"] = h[0].difficulty
            metrics["HALSTEAD_EFFORT"] = h[0].effort

    except SyntaxError:
        pass  

    return metrics


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

        # DEFAULT VALUE
        total_loc = None
        h_volume = None
        h_difficulty = None
        num_operators = None
        num_operands = None

        # RADON → PYTHON ONLY
        if is_python_file(file_path):
            try:
                raw = analyze(code)
                halstead = h_visit(code)

                total_loc = raw.loc

                if halstead:
                    h_volume = halstead[0].volume
                    h_difficulty = halstead[0].difficulty
                    num_operators = halstead[0].N1
                    num_operands = halstead[0].N2

            except SyntaxError:
                pass  

        # LIZARD MULTI LANGUAGE
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

        THRESHOLD = 0.5
        proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (proba >= THRESHOLD).astype(int)

        def risk_label(prob):
            if prob < 0.4:
                return "Low Risk"
            elif prob <= 0.6:
                return "Medium Risk"
            else:
                return "High Risk"
        
        df_result = df_metrics.copy()

        df_result["defect_probability"] = proba
        df_result["prediction"] = y_pred
        df_result["risk_level"] = df_result["defect_probability"].apply(risk_label)

        threshold = st.slider(
            "Defect Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )

        df_result["prediction"] = (df_result["defect_probability"] >= threshold).astype(int)
        df_result["risk_level"] = df_result["defect_probability"].apply(risk_label)

        st.dataframe(df_result)