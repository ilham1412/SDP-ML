import lizard
import pandas as pd

FEATURE_COLUMNS = [
    "LOC_EXECUTABLE",
    "CYCLOMATIC_COMPLEXITY",
    "NUM_PARAMETERS"
]

def extract_metrics(path):
    data = []
    analysis = lizard.analyze_path(path)

    for func in analysis.function_list:
        data.append({
            "function": func.name,
            "file": func.filename,
            "LOC_EXECUTABLE": func.nloc,
            "CYCLOMATIC_COMPLEXITY": func.cyclomatic_complexity,
            "NUM_PARAMETERS": func.parameters
        })

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("Tidak ada function terdeteksi")

    return df
