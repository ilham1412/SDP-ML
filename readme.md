# Software Defect Prediction (SDP)

A machine learning-based application for predicting software defects by analyzing source code metrics. This project uses the NASA JM1 dataset to train models that can identify potentially defective code functions.

##  Overview

Software Defect Prediction (SDP) helps developers identify code that is likely to contain bugs before testing. By analyzing code metrics such as cyclomatic complexity and lines of code, the system provides risk assessments for individual functions.

##  Features

- **Multi-language Support**: Analyze source code in Python, Java, C, and C++
- **Code Metrics Extraction**: Automatically extracts metrics using Lizard and Radon libraries
- **Risk Level Classification**: Categorizes functions into Low, Medium, and High risk
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Adjustable Threshold**: Customize defect prediction sensitivity

##  Metrics Analyzed

### Basic Model (2 Features)
| Metric | Description |
|--------|-------------|
| `LOC_EXECUTABLE` | Lines of executable code |
| `CYCLOMATIC_COMPLEXITY` | Code complexity measure |

### Extended Model (7 Features)
| Metric | Description |
|--------|-------------|
| `LOC_EXECUTABLE` | Lines of executable code |
| `LOC_TOTAL` | Total lines of code |
| `CYCLOMATIC_COMPLEXITY` | Code complexity measure |
| `HALSTEAD_VOLUME` | Program vocabulary size |
| `HALSTEAD_DIFFICULTY` | Difficulty to understand code |
| `NUM_OPERATORS` | Number of operators |
| `NUM_OPERANDS` | Number of operands |

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SDP-Project.git
   cd SDP-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Additional dependencies for the web app**
   ```bash
   pip install streamlit lizard radon
   ```

##  Project Structure

```
SDP-Project/
 app.py                 # Streamlit app (basic 2-feature model)
 app2.py                # Streamlit app (extended 7-feature model)
 train.ipynb            # Model training notebook
 train2.ipynb           # Extended model training notebook
 metric_extractor.py    # Code metrics extraction utility
 JM1.arff               # NASA JM1 dataset
 requirement.txt        # Python dependencies
 readme.md              # Project documentation
 model/                 # Trained models
     final_model.pkl
     scaler_final.pkl
     rf_7_features.pkl
     scaler_7_features.pkl
     ...
```

##  Usage

### Running the Web Application

**Basic Model (2 features):**
```bash
streamlit run app.py
```

**Extended Model (7 features):**
```bash
streamlit run app2.py
```

### How to Use

1. Launch the Streamlit application
2. Upload a source code file (.py, .java, .c, or .cpp)
3. View extracted metrics for each function
4. Review defect predictions and risk levels
5. Adjust the threshold slider to customize sensitivity

##  Model Details

### Dataset
- **Source**: NASA JM1 Dataset (PROMISE Repository)
- **Samples**: 7,747 modules
- **Features**: 21 software metrics
- **Target**: Binary classification (Defective/Non-defective)

### Algorithm
- **Model**: Random Forest Classifier
- **Hyperparameters**:
  - `n_estimators`: 300
  - `class_weight`: {0:1, 1:3} (handling imbalanced data)
- **Preprocessing**: 
  - SMOTE for oversampling minority class
  - StandardScaler for feature normalization

### Training Pipeline
1. Load JM1 dataset
2. Split data (80% train, 20% test) with stratification
3. Apply SMOTE to balance classes
4. Scale features using StandardScaler
5. Train Random Forest with GridSearchCV
6. Evaluate using ROC-AUC and classification metrics

##  Risk Classification

| Probability | Risk Level |
|-------------|------------|
| < 0.4 |  Low Risk |
| 0.4 - 0.6 |  Medium Risk |
| > 0.6 |  High Risk |

##  Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
streamlit
lizard
radon
scipy
```

##  Technologies Used

- **Lizard**: Multi-language code complexity analyzer
- **Radon**: Python-specific metrics (Halstead metrics)
- **Scikit-learn**: Machine learning algorithms
- **Imbalanced-learn**: SMOTE oversampling
- **Streamlit**: Web application framework

##  License

This project is open source and available under the [MIT License](LICENSE).

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  Contact

For questions or feedback, please open an issue in the repository.

---

 If you find this project useful, please give it a star!
