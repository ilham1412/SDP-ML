import streamlit as st
import tempfile
import os
import joblib
import lizard
import pandas as pd

FEATURE_COLUMNS = [
    "LOC_EXECUTABLE",
    "CYCLOMATIC_COMPLEXITY"
]

def extract_metrics(path):
    data = []
    # Get all source files in the directory
    files = [os.path.join(path, f) for f in os.listdir(path) 
             if f.endswith(('.py', '.java', '.c', '.cpp'))]
    
    for file_info in lizard.analyze_files(files):
        for func in file_info.function_list:
            data.append({
                "function": func.name,
                "file": file_info.filename,
                "LOC_EXECUTABLE": func.nloc,
                "CYCLOMATIC_COMPLEXITY": func.cyclomatic_complexity,
                "NUM_PARAMETERS": len(func.parameters)
            })

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("Tidak ada function terdeteksi")

    return df

st.title("Software Defect Prediction")

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
        st.write("Extracted Metrics", df_metrics)

        model = joblib.load("final_model.pkl")
        scaler = joblib.load("scaler_final.pkl")

        FEATURES = ["LOC_EXECUTABLE", "CYCLOMATIC_COMPLEXITY"]

        X = df_metrics[FEATURES]
        X_scaled = scaler.transform(X)

        proba = model.predict_proba(X_scaled)[:, 1]
        df_metrics["Defect_Probability"] = proba
        df_metrics["Defect_Prediction"] = (proba >= 0.3).astype(int)

        st.write("Prediction Result", df_metrics)
