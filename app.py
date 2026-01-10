import streamlit as st
import tempfile
import os
import joblib
from metrics_extractor import extract_metrics

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

        model = joblib.load("defect_model_final.pkl")
        scaler = joblib.load("scaler.pkl")

        FEATURES = ["LOC_EXECUTABLE", "CYCLOMATIC_COMPLEXITY", "NUM_PARAMETERS"]

        X = df_metrics[FEATURES]
        X_scaled = scaler.transform(X)

        proba = model.predict_proba(X_scaled)[:, 1]
        df_metrics["Defect_Probability"] = proba
        df_metrics["Defect_Prediction"] = (proba >= 0.3).astype(int)

        st.write("Prediction Result", df_metrics)
