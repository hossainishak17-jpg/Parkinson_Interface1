
import os
import hashlib
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

warnings.filterwarnings("ignore")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Parkinson's Voice Classification",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CLEAN WHITE INTERFACE
# =========================================================

st.markdown("""
<style>

/* Main application */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #ffffff !important;
    color: #111111 !important;
}

/* Main content */
.main .block-container {
    background-color: #ffffff !important;
    color: #111111 !important;
    padding-top: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f1f4f8 !important;
    border-right: 1px solid #d5dce5 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #111111 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #111111 !important;
    font-weight: 700 !important;
}

/* General text */
.stApp p,
.stApp label,
.stApp li,
.stApp small {
    color: #111111 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #111111 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 9px !important;
}

[data-testid="stExpander"] details {
    background-color: #f8fafc !important;
}

[data-testid="stExpander"] summary {
    background-color: #e8eef5 !important;
    color: #111111 !important;
}

[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #111111 !important;
}

[data-testid="stExpander"] details > div {
    background-color: #f8fafc !important;
    color: #111111 !important;
}

/* Dataframes and tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #d5dce5 !important;
    border-radius: 7px !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 6px !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #94a3b8 !important;
}

[data-baseweb="select"] span {
    color: #111111 !important;
}

/* Dropdown options */
[role="listbox"],
[role="option"] {
    background-color: #ffffff !important;
    color: #111111 !important;
}

[role="option"]:hover {
    background-color: #e2e8f0 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 9px !important;
    padding: 12px !important;
}

[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    border: 1px dashed #94a3b8 !important;
    border-radius: 7px !important;
}

[data-testid="stFileUploader"] * {
    color: #111111 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: #e2e8f0 !important;
    color: #111111 !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #cbd5e1 !important;
    color: #000000 !important;
    border-color: #64748b !important;
}

/* Radio buttons and checkboxes */
[role="radiogroup"] label,
[data-testid="stCheckbox"] label {
    color: #111111 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background-color: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #111111 !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #111111 !important;
}

/* Captions */
[data-testid="stCaptionContainer"],
.stCaption {
    color: #475569 !important;
}

/* Slider */
[data-testid="stSlider"] {
    color: #111111 !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #111111 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1d4ed8 !important;
}

/* Progress bar */
[data-testid="stProgress"] {
    background-color: #e2e8f0 !important;
}

/* Dividers */
hr {
    border-color: #cbd5e1 !important;
}

/* Code blocks */
.stCode,
pre,
code {
    background-color: #f1f5f9 !important;
    color: #111111 !important;
}

/* Links */
a {
    color: #1d4ed8 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "data_hash": None,
    "X": None,
    "y": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "models": {},
    "results": None,
    "trained": False,
    "training_error": None,
    "shap_result": None,
    "shap_model": None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_training():

    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None

    st.session_state.models = {}
    st.session_state.results = None

    st.session_state.trained = False
    st.session_state.training_error = None

    st.session_state.shap_result = None
    st.session_state.shap_model = None


# =========================================================
# DATASET COLUMN NAMES
# =========================================================

UCI_COLUMNS = [
    "name",
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "status",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(uploaded_file):

    if uploaded_file is not None:

        uploaded_file.seek(0)
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, header=None)

    elif os.path.exists("parkinsons.csv"):

        df = pd.read_csv("parkinsons.csv")

    elif os.path.exists("parkinsons.data"):

        df = pd.read_csv("parkinsons.data", header=None)

    else:
        return None

    df.columns = [
        str(c).strip().strip('"').strip("'")
        for c in df.columns
    ]

    lower_columns = [c.lower() for c in df.columns]

    if "status" not in lower_columns and df.shape[1] == 24:
        df.columns = UCI_COLUMNS

    return df


# =========================================================
# PREPARE DATASET
# =========================================================

def prepare_dataset(df):

    data = df.copy()

    data.columns = [str(c).strip() for c in data.columns]

    target_col = next(
        (c for c in data.columns if c.lower() == "status"),
        None
    )

    if target_col is None:
        raise ValueError("Target column 'status' was not found.")

    y = pd.to_numeric(data[target_col], errors="coerce")

    valid_rows = y.notna()

    data = data.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Status must contain only 0 and 1.")

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "Both Healthy (0) and Parkinson's (1) classes are required."
        )

    excluded = {
        "status", "name", "subject_id",
        "subject", "patient_id", "id"
    }

    feature_columns = [
        c for c in data.columns
        if c.lower() not in excluded
    ]

    X = data[feature_columns].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, ~X.columns.duplicated()]

    if X.empty:
        raise ValueError("No valid numeric features found.")

    if y.value_counts().min() < 2:
        raise ValueError("Each class needs at least two records.")

    return X, y


# =========================================================
# CREATE MODELS
# =========================================================

def create_models(minority_count, train_count):

    cv_splits = min(5, int(minority_count))

    if cv_splits < 2:
        raise ValueError("Not enough samples for Stacking.")

    knn_neighbors = min(3, int(train_count))

    base_models = [

        (
            "lr",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=42
                ))
            ])
        ),

        (
            "svm",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(
                    probability=True,
                    random_state=42
                ))
            ])
        ),

        (
            "knn",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(
                    n_neighbors=knn_neighbors
                ))
            ])
        ),

        (
            "dt",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(
                    random_state=42,
                    class_weight="balanced"
                ))
            ])
        ),

        (
            "rf",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                ))
            ])
        )
    ]

    models = {

        "Logistic Regression": clone(base_models[0][1]),
        "SVM": clone(base_models[1][1]),
        "KNN": clone(base_models[2][1]),
        "Decision Tree": clone(base_models[3][1]),
        "Random Forest": clone(base_models[4][1]),

        "Hard Voting": VotingClassifier(
            estimators=base_models,
            voting="hard"
        ),

        "Soft Voting": VotingClassifier(
            estimators=base_models,
            voting="soft"
        ),

        "Stacking": StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=2000),
            cv=StratifiedKFold(
                n_splits=cv_splits,
                shuffle=True,
                random_state=42
            ),
            stack_method="predict_proba"
        )
    }

    return models


# =========================================================
# GET POSITIVE CLASS PROBABILITY
# =========================================================

def get_probability(model, X):

    try:

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)
            classes = list(model.classes_)

            if 1 in classes:
                return probabilities[:, classes.index(1)]

        if hasattr(model, "decision_function"):

            scores = np.asarray(
                model.decision_function(X),
                dtype=float
            )

            if scores.ndim == 1:

                scores = np.clip(scores, -500, 500)

                return 1 / (1 + np.exp(-scores))

    except Exception:
        return None

    return None


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = get_probability(model, X_test)

    auc = np.nan

    if probabilities is not None and len(np.unique(y_test)) == 2:

        try:
            auc = roc_auc_score(y_test, probabilities)
        except Exception:
            pass

    return {

        "Accuracy": accuracy_score(y_test, predictions),

        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),

        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),

        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),

        "ROC-AUC": auc,

        "MCC": matthews_corrcoef(y_test, predictions)
    }


# =========================================================
# TRAIN ALL MODELS
# =========================================================

def train_all_models(X_train, X_test, y_train, y_test):

    models = create_models(
        y_train.value_counts().min(),
        len(X_train)
    )

    trained_models = {}
    result_rows = []

    progress = st.progress(0)
    status = st.empty()

    total = len(models)

    for i, (name, model) in enumerate(models.items()):

        status.write(f"Training {name}...")

        model.fit(X_train, y_train)

        metrics = calculate_metrics(
            model, X_test, y_test
        )

        trained_models[name] = model

        result_rows.append({
            "Model": name,
            **metrics
        })

        progress.progress((i + 1) / total)

    status.success("All models trained successfully.")

    return trained_models, pd.DataFrame(result_rows)


# =========================================================
# TESTING DATA PREPARATION
# =========================================================

def prepare_testing_csv(df, feature_names):

    data = df.copy()

    data.columns = [str(c).strip() for c in data.columns]

    actual_status = None

    status_col = next(
        (c for c in data.columns if c.lower() == "status"),
        None
    )

    if status_col is not None:

        actual_status = pd.to_numeric(
            data[status_col], errors="coerce"
        )

        if actual_status.isna().any():
            raise ValueError(
                "Testing status contains missing or invalid labels."
            )

        actual_status = actual_status.astype(int)

        if not set(actual_status.unique()).issubset({0, 1}):
            raise ValueError("Testing status must contain only 0 and 1.")

    ignored = {
        "status", "name", "subject_id",
        "subject", "patient_id", "id"
    }

    feature_data = data[
        [c for c in data.columns if c.lower() not in ignored]
    ].copy()

    missing = [
        c for c in feature_names
        if c not in feature_data.columns
    ]

    if missing:
        raise ValueError("Missing features: " + ", ".join(missing))

    feature_data = feature_data[feature_names].copy()

    for col in feature_names:
        feature_data[col] = pd.to_numeric(
            feature_data[col], errors="coerce"
        )

    if feature_data.isna().all(axis=1).any():
        raise ValueError(
            "Some testing rows contain no valid numeric features."
        )

    return feature_data, actual_status


# =========================================================
# DOWNLOAD CSV
# =========================================================

def download_csv(df, filename, key):

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Parkinson's Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Prediction",
        "Model Comparison",
        "SHAP Explainability",
        "Model Evaluation"
    ]
)

uploaded_dataset = st.sidebar.file_uploader(
    "Training Dataset",
    type=["csv", "data", "txt"],
    key="training_dataset"
)


# =========================================================
# LOAD DATA
# =========================================================

try:

    raw_data = load_dataset(uploaded_dataset)

    if raw_data is None:

        st.error(
            "Upload your Parkinson's dataset or place "
            "parkinsons.csv beside app.py."
        )

        st.stop()

    current_hash = hashlib.md5(
        pd.util.hash_pandas_object(
            raw_data,
            index=True
        ).values.tobytes()
    ).hexdigest()

    if st.session_state.data_hash != current_hash:

        X, y = prepare_dataset(raw_data)

        st.session_state.data_hash = current_hash
        st.session_state.X = X.copy()
        st.session_state.y = y.copy()

        reset_training()

    else:

        X = st.session_state.X
        y = st.session_state.y

except Exception as e:

    st.error(f"Dataset Error: {e}")
    st.stop()


# =========================================================
# TRAINING SETTINGS
# =========================================================

st.sidebar.divider()

test_percent = st.sidebar.slider(
    "Internal Test Size (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

if st.sidebar.button(
    "Train All 8 Models",
    use_container_width=True
):

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_percent / 100,
            random_state=42,
            stratify=y
        )

        with st.spinner("Training all models..."):

            models, results = train_all_models(
                X_train, X_test, y_train, y_test
            )

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test

        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.session_state.models = models
        st.session_state.results = results

        st.session_state.trained = True
        st.session_state.training_error = None

        st.session_state.shap_result = None
        st.session_state.shap_model = None

        st.sidebar.success("Training completed.")

    except Exception as e:

        reset_training()

        st.session_state.training_error = str(e)

        st.sidebar.error("Training failed.")

if st.session_state.training_error:
    st.error(st.session_state.training_error)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("Parkinson's Disease Voice Classification")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Recordings", len(raw_data))
    col2.metric("Features", X.shape[1])
    col3.metric("Healthy", int((y == 0).sum()))
    col4.metric("Parkinson's", int((y == 1).sum()))

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        raw_data.head(8),
        use_container_width=True
    )

    st.subheader("Class Distribution")

    counts = y.value_counts().reindex(
        [0, 1], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(
        ["Healthy", "Parkinson's"],
        counts.values
    )

    ax.set_ylabel("Recordings")
    ax.set_xlabel("Class")
    ax.set_title("Dataset Class Distribution")

    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    st.pyplot(fig)
    plt.close(fig)

    if st.session_state.trained:
        st.success("Models are trained and ready.")
    else:
        st.info("Train all models from the sidebar.")


# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":

    st.title("Prediction")

    if not st.session_state.trained:
        st.warning("Train all models first.")
        st.stop()

    features = list(st.session_state.X_train.columns)

    st.subheader("Upload Testing Sets")

    st.caption(
        "Upload up to 5 testing CSV files. "
        "All 8 models will predict each uploaded dataset."
    )

    testing_files = st.file_uploader(
        "Testing CSV Files",
        type=["csv"],
        accept_multiple_files=True,
        key="multiple_testing_files"
    )

    if testing_files:

        if len(testing_files) > 5:

            st.error("Maximum 5 testing files are allowed.")

        else:

            for file_index, testing_file in enumerate(testing_files):

                with st.expander(
                    testing_file.name,
                    expanded=True
                ):

                    try:

                        testing_file.seek(0)

                        testing_df = pd.read_csv(testing_file)

                        X_external, actual_status = prepare_testing_csv(
                            testing_df, features
                        )

                        st.write(
                            f"Testing Records: {len(X_external)}"
                        )

                        model_outputs = []

                        for model_name, model in st.session_state.models.items():

                            predictions = model.predict(X_external)

                            probabilities = get_probability(
                                model, X_external
                            )

                            output = pd.DataFrame({

                                "Row": np.arange(
                                    1, len(X_external) + 1
                                ),

                                "Model": model_name,

                                "Prediction": predictions,

                                "Prediction_Label": [
                                    "Parkinson's" if p == 1 else "Healthy"
                                    for p in predictions
                                ]
                            })

                            if probabilities is not None:

                                output["Parkinsons_Probability"] = probabilities

                                output["Confidence_Score"] = np.where(
                                    predictions == 1,
                                    probabilities,
                                    1 - probabilities
                                )

                            if actual_status is not None:

                                actual = actual_status.to_numpy()

                                output["Actual_Status"] = actual

                                output["Actual_Label"] = [
                                    "Parkinson's" if p == 1 else "Healthy"
                                    for p in actual
                                ]

                                output["Correct"] = (
                                    predictions == actual
                                )

                                metrics = calculate_metrics(
                                    model, X_external, actual
                                )

                                st.write(f"**{model_name}**")

                                m1, m2, m3 = st.columns(3)

                                m1.metric(
                                    "Accuracy",
                                    f"{metrics['Accuracy']:.4f}"
                                )

                                m2.metric(
                                    "F1-Score",
                                    f"{metrics['F1-Score']:.4f}"
                                )

                                m3.metric(
                                    "Recall",
                                    f"{metrics['Recall']:.4f}"
                                )

                            model_outputs.append(output)

                        combined_output = pd.concat(
                            model_outputs,
                            ignore_index=True
                        )

                        st.dataframe(
                            combined_output,
                            use_container_width=True
                        )

                        safe_name = os.path.splitext(
                            testing_file.name
                        )[0]

                        download_csv(
                            combined_output,
                            f"{safe_name}_predictions.csv",
                            f"prediction_download_{file_index}"
                        )

                    except Exception as e:

                        st.error(f"Testing Error: {e}")

    else:

        st.info(
            "Upload testing CSV files to generate predictions."
        )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    if not st.session_state.trained:
        st.warning("Train all models first.")
        st.stop()

    results = st.session_state.results.copy()

    selected_model = st.selectbox(
        "Select Model",
        results["Model"].tolist()
    )

    selected_row = results[
        results["Model"] == selected_model
    ].iloc[0]

    st.subheader(selected_model)

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{selected_row['Accuracy']:.4f}")
    c2.metric("Precision", f"{selected_row['Precision']:.4f}")
    c3.metric("Recall", f"{selected_row['Recall']:.4f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("F1-Score", f"{selected_row['F1-Score']:.4f}")

    c5.metric(
        "ROC-AUC",
        "N/A" if pd.isna(selected_row["ROC-AUC"])
        else f"{selected_row['ROC-AUC']:.4f}"
    )

    c6.metric("MCC", f"{selected_row['MCC']:.4f}")

    st.divider()

    selected_metric = st.selectbox(
        "Compare Models By",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "MCC"
        ]
    )

    chart_data = results.sort_values(
        selected_metric,
        ascending=False,
        na_position="last"
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        chart_data["Model"],
        chart_data[selected_metric].fillna(0)
    )

    ax.set_ylabel(selected_metric)
    ax.set_xlabel("Models")
    ax.set_title(f"Model Comparison - {selected_metric}")

    ax.tick_params(
        axis="x",
        rotation=35,
        colors="black"
    )

    ax.tick_params(axis="y", colors="black")

    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.title.set_color("black")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Selected Model Result")

    st.dataframe(
        results[results["Model"] == selected_model].set_index("Model"),
        use_container_width=True
    )

    with st.expander("All Models Results"):

        st.dataframe(
            results,
            use_container_width=True
        )

    download_csv(
        results,
        "model_comparison_results.csv",
        "comparison_download"
    )


# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

elif page == "SHAP Explainability":

    st.title("SHAP Explainability")

    if not st.session_state.trained:
        st.warning("Train all models first.")
        st.stop()

    selected_name = st.selectbox(
        "Select Model for SHAP",
        list(st.session_state.models.keys())
    )

    st.caption(
        "SHAP explains the contribution of each voice feature "
        "to the selected model's output."
    )

    if st.button("Calculate SHAP Values"):

        try:

            import shap

            model = st.session_state.models[selected_name]

            X_train = st.session_state.X_train
            X_test = st.session_state.X_test

            features = list(X_train.columns)

            background_df = X_train.sample(
                min(20, len(X_train)),
                random_state=42
            )

            explain_df = X_test.iloc[
                :min(25, len(X_test))
            ].copy()

            background = background_df.to_numpy(dtype=float)
            explain_data = explain_df.to_numpy(dtype=float)

            def prediction_function(values):

                values = np.asarray(values, dtype=float)

                input_df = pd.DataFrame(
                    values,
                    columns=features
                )

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_df)

                    classes = list(model.classes_)

                    if 1 in classes:
                        return probabilities[:, classes.index(1)]

                if hasattr(model, "decision_function"):

                    scores = np.asarray(
                        model.decision_function(input_df)
                    )

                    if scores.ndim == 1:

                        scores = np.clip(scores, -500, 500)

                        return 1 / (1 + np.exp(-scores))

                return model.predict(input_df).astype(float)

            with st.spinner("Calculating SHAP values..."):

                explainer = shap.KernelExplainer(
                    prediction_function,
                    background
                )

                raw_values = explainer.shap_values(
                    explain_data,
                    nsamples=100
                )

            if isinstance(raw_values, list):

                shap_values = np.asarray(raw_values[-1])

            else:

                shap_values = np.asarray(raw_values)

                if shap_values.ndim == 3:

                    if shap_values.shape[-1] == 2:
                        shap_values = shap_values[:, :, 1]

                    elif shap_values.shape[0] == 2:
                        shap_values = shap_values[1]

            shap_values = np.squeeze(shap_values)

            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            if shap_values.shape != explain_data.shape:

                raise ValueError(
                    f"Unexpected SHAP shape: {shap_values.shape}. "
                    f"Expected: {explain_data.shape}"
                )

            importance = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame({

                "Feature": features,
                "Mean Absolute SHAP": importance

            }).sort_values(
                "Mean Absolute SHAP",
                ascending=False
            )

            st.session_state.shap_result = (
                shap_values,
                explain_data,
                importance_df
            )

            st.session_state.shap_model = selected_name

        except ImportError:

            st.error("SHAP is not installed. Run: pip install shap")

        except Exception as e:

            st.error(f"SHAP Error: {e}")

    if (
        st.session_state.shap_result is not None
        and st.session_state.shap_model == selected_name
    ):

        shap_values, explain_data, importance_df = (
            st.session_state.shap_result
        )

        st.subheader(
            f"{selected_name} - Feature Importance"
        )

        top_n = st.slider(
            "Number of Features",
            min_value=1,
            max_value=len(importance_df),
            value=min(15, len(importance_df)),
            key="shap_top_n"
        )

        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.barh(
            top_features["Feature"][::-1],
            top_features["Mean Absolute SHAP"][::-1]
        )

        ax.set_xlabel("Mean Absolute SHAP Value")
        ax.set_title("SHAP Feature Importance")

        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.title.set_color("black")

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("SHAP Summary Plot")

        try:

            shap.summary_plot(
                shap_values,
                explain_data,
                feature_names=list(
                    st.session_state.X_train.columns
                ),
                show=False
            )

            fig = plt.gcf()
            fig.patch.set_facecolor("white")

            for ax in fig.axes:

                ax.set_facecolor("white")
                ax.tick_params(colors="black")

                ax.xaxis.label.set_color("black")
                ax.yaxis.label.set_color("black")
                ax.title.set_color("black")

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        except Exception as e:

            st.warning(f"Summary plot unavailable: {e}")

        st.subheader("Feature Importance Table")

        st.dataframe(
            importance_df,
            use_container_width=True
        )

        download_csv(
            importance_df,
            "shap_feature_importance.csv",
            "shap_download"
        )


# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    if not st.session_state.trained:
        st.warning("Train all models first.")
        st.stop()

    selected_name = st.selectbox(
        "Select Model",
        list(st.session_state.models.keys())
    )

    model = st.session_state.models[selected_name]

    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    predictions = model.predict(X_test)
    probabilities = get_probability(model, X_test)

    metrics = calculate_metrics(
        model, X_test, y_test
    )

    st.subheader("Evaluation Metrics")

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "MCC"
    ]

    cols = st.columns(3)

    for i, name in enumerate(metric_names):

        value = metrics[name]

        display = (
            "N/A" if pd.isna(value)
            else f"{value:.4f}"
        )

        cols[i % 3].metric(name, display)

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.imshow(cm, cmap="Blues")

    ax.set_xticks(
        [0, 1],
        labels=["Healthy", "Parkinson's"]
    )

    ax.set_yticks(
        [0, 1],
        labels=["Healthy", "Parkinson's"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    ax.tick_params(colors="black")

    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Healthy", "Parkinson's"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    if probabilities is not None:

        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        auc_value = roc_auc_score(
            y_test,
            probabilities
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {auc_value:.4f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        ax.tick_params(colors="black")

        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.info(
            "ROC Curve is not available for this model."
        )
