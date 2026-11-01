import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Parkinson's Disease Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #071426 0%, #101d35 55%, #17233d 100%);
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background: #0B1528;
    border-right: 1px solid #263954;
}

h1, h2, h3 {
    color: #E2E8F0 !important;
}

p, label, span {
    color: #CBD5E1;
}

.stButton > button {
    background: #0284C7;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #0369A1;
    color: white;
}

div[data-testid="stMetric"] {
    background: #14243D;
    border: 1px solid #29415F;
    padding: 16px;
    border-radius: 10px;
}

div[data-testid="stMetric"] label {
    color: #94A3B8 !important;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #14243D;
    color: white;
}

.stAlert {
    border-radius: 8px;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

default_states = {
    "models": {},
    "subject_models": {},
    "metrics": None,
    "subject_metrics": None,
    "latest_input": None,
    "latest_comparison": None,
    "batch_result": None,
    "feature_columns": None,
    "dataset": None,
    "dataset_signature": None,
    "training_done": False,
    "subject_split_info": None
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# DATASET FUNCTIONS
# =========================================================

def load_dataset(uploaded_file):

    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Unable to read uploaded dataset: {e}")
            return None

    base_dir = os.path.dirname(os.path.abspath(__file__))

    filenames = [
        "parkinsons.csv",
        "parkinsons.data",
        "parkinson.data"
    ]

    for filename in filenames:
        file_path = os.path.join(base_dir, filename)

        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Unable to read {filename}: {e}")
                return None

    return None


def prepare_dataset(data):

    df = data.copy()

    df.columns = df.columns.astype(str).str.strip()

    if "status" not in df.columns:
        raise ValueError(
            "The dataset must contain a target column named 'status'."
        )

    y = pd.to_numeric(df["status"], errors="coerce")

    valid_rows = y.notna()

    df = df.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    excluded_columns = [
        "status",
        "name",
        "subject_id"
    ]

    feature_columns = []

    for column in df.columns:

        if column in excluded_columns:
            continue

        converted = pd.to_numeric(df[column], errors="coerce")

        if converted.notna().any():
            df[column] = converted
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No numerical feature columns were found.")

    X = df[feature_columns].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    valid = y.isin([0, 1])

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    df = df.loc[valid].reset_index(drop=True)

    if y.nunique() != 2:
        raise ValueError(
            "The target column must contain both classes: 0 and 1."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each target class must contain at least two records."
        )

    return df, X, y, feature_columns


# =========================================================
# MODEL FUNCTIONS
# =========================================================

def create_models(training_size, minority_count):

    neighbors = max(1, min(5, training_size))

    models = {

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=42
            ))
        ]),

        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            ))
        ]),

        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=neighbors
            ))
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
                class_weight="balanced"
            ))
        ]),

        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
                class_weight="balanced"
            ))
        ]),

        "AdaBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", AdaBoostClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
    }

    # Optional XGBoost

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            ))
        ])

    except Exception:
        pass

    base_names = list(models.keys())

    if minority_count >= 2:

        estimators = [
            (name.replace(" ", "_"), models[name])
            for name in base_names
            if name != "XGBoost"
        ]

        if len(estimators) >= 2:

            models["Hard Voting"] = VotingClassifier(
                estimators=estimators,
                voting="hard",
                n_jobs=1
            )

            models["Soft Voting"] = VotingClassifier(
                estimators=estimators,
                voting="soft",
                n_jobs=1
            )

            cv = min(3, minority_count)

            models["Stacking"] = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(
                    max_iter=1000
                ),
                cv=cv,
                n_jobs=1
            )

    return models


def get_probability(model, X):

    try:

        if not hasattr(model, "predict_proba"):
            return None

        probabilities = model.predict_proba(X)

        classes = list(model.classes_)

        if 1 in classes:
            index = classes.index(1)
            return probabilities[:, index]

        return None

    except Exception:
        return None


def calculate_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)

    probability = get_probability(model, X_test)

    result = {
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
        "MCC": matthews_corrcoef(y_test, predictions)
    }

    if probability is not None and len(np.unique(y_test)) == 2:
        result["ROC-AUC"] = roc_auc_score(
            y_test,
            probability
        )
    else:
        result["ROC-AUC"] = np.nan

    return result


# =========================================================
# TRAINING FUNCTIONS
# =========================================================

def train_random_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = create_models(
        len(X_train),
        int(y_train.value_counts().min())
    )

    trained_models = {}
    results = []

    progress = st.progress(0)
    total = len(models)

    for index, (name, model) in enumerate(models.items()):

        try:

            model.fit(X_train, y_train)

            trained_models[name] = model

            result = calculate_metrics(
                model,
                X_test,
                y_test
            )

            result["Model"] = name
            results.append(result)

        except Exception as e:

            st.warning(
                f"{name} could not be trained: {str(e)}"
            )

        progress.progress((index + 1) / total)

    progress.empty()

    if not trained_models:
        raise ValueError("No machine learning model could be trained.")

    metrics_df = pd.DataFrame(results)

    return (
        trained_models,
        metrics_df,
        X_train,
        X_test,
        y_train,
        y_test
    )


def create_subject_split(df, X, y):

    if "name" not in df.columns:
        raise ValueError(
            "Subject-disjoint evaluation requires a 'name' column."
        )

    groups = df["name"].astype(str).str.extract(
        r"(S\d+)",
        expand=False
    )

    if groups.isna().any():
        raise ValueError(
            "Subject IDs could not be extracted from the name column."
        )

    if groups.nunique() < 2:
        raise ValueError(
            "At least two different subjects are required."
        )

    best_split = None
    best_score = float("inf")

    splitter = GroupShuffleSplit(
        n_splits=500,
        test_size=0.20,
        random_state=42
    )

    overall_ratio = y.mean()

    for train_idx, test_idx in splitter.split(
        X,
        y,
        groups
    ):

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() != 2:
            continue

        if y_test.nunique() != 2:
            continue

        test_size_difference = abs(
            len(test_idx) / len(y) - 0.20
        )

        class_ratio_difference = (
            abs(y_train.mean() - overall_ratio)
            + abs(y_test.mean() - overall_ratio)
        )

        score = (
            test_size_difference
            + class_ratio_difference
        )

        if score < best_score:

            best_score = score
            best_split = (train_idx, test_idx)

    if best_split is None:
        raise ValueError(
            "A valid subject-disjoint split could not be created."
        )

    train_idx, test_idx = best_split

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise ValueError(
            "Subject overlap detected. Evaluation stopped."
        )

    return (
        train_idx,
        test_idx,
        groups,
        train_groups,
        test_groups
    )


def train_subject_models(df, X, y):

    (
        train_idx,
        test_idx,
        groups,
        train_groups,
        test_groups
    ) = create_subject_split(df, X, y)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    models = create_models(
        len(X_train),
        int(y_train.value_counts().min())
    )

    trained_models = {}
    results = []

    for name, model in models.items():

        try:

            model.fit(X_train, y_train)

            trained_models[name] = model

            result = calculate_metrics(
                model,
                X_test,
                y_test
            )

            result["Model"] = name
            results.append(result)

        except Exception as e:

            st.warning(
                f"{name} failed during subject-disjoint training: {e}"
            )

    if not trained_models:
        raise ValueError(
            "No model could be trained for subject-disjoint evaluation."
        )

    return (
        trained_models,
        pd.DataFrame(results),
        train_groups,
        test_groups,
        len(train_idx),
        len(test_idx)
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Parkinson's AI System")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "data"]
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Manual Voice Input",
        "All Models Comparison",
        "Model Evaluation",
        "Subject-Disjoint Evaluation",
        "Batch Prediction",
        "SHAP Explainability",
        "Model Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Parkinson's Disease Voice Classification"
)


# =========================================================
# LOAD DATA
# =========================================================

raw_data = load_dataset(uploaded_file)

if raw_data is None:

    st.title("Parkinson's Disease Classification")

    st.warning(
        "Dataset not found. Upload a CSV file from the sidebar "
        "or place parkinsons.csv in the same folder as app.py."
    )

    st.stop()


try:

    df, X, y, feature_columns = prepare_dataset(raw_data)

except Exception as e:

    st.error(f"Dataset error: {e}")
    st.stop()


# Reset training when dataset changes

signature = (
    tuple(df.columns),
    len(df),
    int(y.sum()),
    int((y == 0).sum())
)

if st.session_state.dataset_signature != signature:

    st.session_state.models = {}
    st.session_state.subject_models = {}
    st.session_state.metrics = None
    st.session_state.subject_metrics = None
    st.session_state.latest_input = None
    st.session_state.latest_comparison = None
    st.session_state.batch_result = None
    st.session_state.training_done = False
    st.session_state.dataset_signature = signature


st.session_state.dataset = df
st.session_state.feature_columns = feature_columns


# =========================================================
# TRAIN MODELS
# =========================================================

if not st.session_state.training_done:

    with st.spinner("Training machine learning models..."):

        try:

            (
                models,
                metrics,
                X_train,
                X_test,
                y_train,
                y_test
            ) = train_random_models(X, y)

            st.session_state.models = models
            st.session_state.metrics = metrics

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            st.session_state.training_done = True

        except Exception as e:

            st.error(f"Model training failed: {e}")
            st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("Parkinson's Disease Voice Classification")

st.caption(
    "Machine Learning | Ensemble Learning | SHAP Explainability"
)

st.markdown("---")


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader("Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Voice Features", len(feature_columns))
    col3.metric("Parkinson's", int((y == 1).sum()))
    col4.metric("Healthy", int((y == 0).sum()))

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Class Distribution")

        counts = y.value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            ["Healthy", "Parkinson's"],
            [
                counts.get(0, 0),
                counts.get(1, 0)
            ]
        )

        ax.set_ylabel("Number of Records")
        ax.set_title("Dataset Class Distribution")

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Model Performance")

        if st.session_state.metrics is not None:

            display_df = st.session_state.metrics[
                ["Model", "Accuracy", "F1-Score"]
            ].copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# MANUAL VOICE INPUT
# =========================================================

elif page == "Manual Voice Input":

    st.subheader("Manual Voice Input")

    st.caption(
        "Enter numerical voice measurements to generate predictions."
    )

    medians = X.median().fillna(0)

    with st.form("manual_prediction_form"):

        input_values = {}

        columns = st.columns(2)

        for index, feature in enumerate(feature_columns):

            with columns[index % 2]:

                input_values[feature] = st.number_input(
                    feature,
                    value=float(medians[feature]),
                    format="%.6f",
                    key=f"manual_{feature}"
                )

        submitted = st.form_submit_button(
            "Run Prediction",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        predictions = []
        probabilities = []

        for name, model in st.session_state.models.items():

            try:

                prediction = int(
                    model.predict(input_df)[0]
                )

                probability = get_probability(
                    model,
                    input_df
                )

                confidence = (
                    float(probability[0])
                    if probability is not None
                    else np.nan
                )

                predictions.append({
                    "Model": name,
                    "Prediction": (
                        "Parkinson's Disease"
                        if prediction == 1
                        else "Healthy"
                    ),
                    "Class": prediction,
                    "Confidence": confidence
                })

                probabilities.append(prediction)

            except Exception as e:

                st.warning(
                    f"{name} prediction failed: {e}"
                )

        if predictions:

            comparison_df = pd.DataFrame(predictions)

            vote = (
                1 if sum(probabilities) > len(probabilities) / 2
                else 0
            )

            st.session_state.latest_input = {
                "prediction": vote,
                "input": input_values
            }

            st.session_state.latest_comparison = comparison_df

            st.success("Prediction completed successfully.")

            if vote == 1:

                st.error("Overall Model Agreement: Parkinson's Disease")

            else:

                st.success("Overall Model Agreement: Healthy")

            st.subheader("Individual Model Predictions")

            st.dataframe(
                comparison_df.drop(columns=["Class"]),
                use_container_width=True,
                hide_index=True
            )

            st.metric(
                "Parkinson's Predictions",
                f"{sum(probabilities)} / {len(probabilities)}"
            )


# =========================================================
# ALL MODELS COMPARISON
# =========================================================

elif page == "All Models Comparison":

    st.subheader("All Models Prediction Comparison")

    st.caption(
        "Compare individual predictions and model agreement."
    )

    comparison = st.session_state.latest_comparison

    if comparison is not None and not comparison.empty:

        display_df = comparison.drop(
            columns=["Class"],
            errors="ignore"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        classes = comparison["Class"].tolist()

        pd_count = classes.count(1)
        healthy_count = classes.count(0)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Models", len(classes))
        col2.metric("Parkinson's Predictions", pd_count)
        col3.metric("Healthy Predictions", healthy_count)

        majority = (
            1 if pd_count > healthy_count
            else 0
        )

        st.markdown("---")

        st.subheader("Overall Model Agreement")

        if majority == 1:
            st.error("Majority Prediction: Parkinson's Disease")
        else:
            st.success("Majority Prediction: Healthy")

    else:

        st.info(
            "No prediction available. Run a prediction from Manual Voice Input."
        )

    st.markdown("---")

    st.subheader("Prediction Result")

    latest = st.session_state.latest_input

    if latest is not None:

        if latest["prediction"] == 1:
            st.error("Parkinson's Disease")
        else:
            st.success("Healthy")

    else:

        st.info("No manual prediction is available.")


# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.subheader("Model Evaluation")

    models = st.session_state.models

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    model = models[selected_model]

    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    predictions = model.predict(X_test)

    probability = get_probability(
        model,
        X_test
    )

    metrics = calculate_metrics(
        model,
        X_test,
        y_test
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.4f}"
    )

    c2.metric(
        "Precision",
        f"{metrics['Precision']:.4f}"
    )

    c3.metric(
        "Recall",
        f"{metrics['Recall']:.4f}"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "F1-Score",
        f"{metrics['F1-Score']:.4f}"
    )

    c5.metric(
        "ROC-AUC",
        (
            f"{metrics['ROC-AUC']:.4f}"
            if not np.isnan(metrics["ROC-AUC"])
            else "N/A"
        )
    )

    c6.metric(
        "MCC",
        f"{metrics['MCC']:.4f}"
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.imshow(cm)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Healthy", "Parkinson's"])
        ax.set_yticklabels(["Healthy", "Parkinson's"])

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            target_names=["Healthy", "Parkinson's"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True
        )

    if probability is not None:

        st.markdown("---")

        st.subheader("ROC Curve")

        fpr, tpr, thresholds = roc_curve(
            y_test,
            probability
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {metrics['ROC-AUC']:.4f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)


# =========================================================
# SUBJECT-DISJOINT EVALUATION
# =========================================================

elif page == "Subject-Disjoint Evaluation":

    st.subheader("Subject-Disjoint Evaluation")

    st.caption(
        "Evaluation using separate subjects in training and testing."
    )

    if "name" not in df.columns:

        st.error(
            "The dataset must contain a 'name' column with subject IDs."
        )

    else:

        if st.button(
            "Run Subject-Disjoint Evaluation",
            use_container_width=True
        ):

            with st.spinner(
                "Training models using subject-disjoint split..."
            ):

                try:

                    (
                        subject_models,
                        subject_metrics,
                        train_groups,
                        test_groups,
                        train_size,
                        test_size
                    ) = train_subject_models(df, X, y)

                    st.session_state.subject_models = subject_models
                    st.session_state.subject_metrics = subject_metrics

                    st.session_state.subject_split_info = {
                        "train_groups": train_groups,
                        "test_groups": test_groups,
                        "train_size": train_size,
                        "test_size": test_size
                    }

                    st.success(
                        "Subject-disjoint evaluation completed."
                    )

                except Exception as e:

                    st.error(str(e))

        if st.session_state.subject_metrics is not None:

            info = st.session_state.subject_split_info

            st.markdown("---")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Training Records",
                info["train_size"]
            )

            c2.metric(
                "Testing Records",
                info["test_size"]
            )

            c3.metric(
                "Subject Overlap",
                len(
                    info["train_groups"].intersection(
                        info["test_groups"]
                    )
                )
            )

            st.subheader("Subject-Disjoint Model Results")

            st.dataframe(
                st.session_state.subject_metrics,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")

            st.subheader("Training and Testing Subjects")

            col1, col2 = st.columns(2)

            with col1:

                st.write("Training Subjects")

                st.write(
                    ", ".join(
                        sorted(info["train_groups"])
                    )
                )

            with col2:

                st.write("Testing Subjects")

                st.write(
                    ", ".join(
                        sorted(info["test_groups"])
                    )
                )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    st.subheader("Batch Prediction")

    st.caption(
        "Upload a CSV file containing voice feature measurements."
    )

    batch_file = st.file_uploader(
        "Upload Prediction CSV",
        type=["csv"],
        key="batch_upload"
    )

    if batch_file is not None:

        try:

            batch_df = pd.read_csv(batch_file)

            missing_features = [
                col for col in feature_columns
                if col not in batch_df.columns
            ]

            if missing_features:

                st.error(
                    "Missing required features: "
                    + ", ".join(missing_features)
                )

            else:

                batch_X = batch_df[feature_columns].copy()

                for column in feature_columns:
                    batch_X[column] = pd.to_numeric(
                        batch_X[column],
                        errors="coerce"
                    )

                batch_X = batch_X.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                if st.button(
                    "Run Batch Prediction",
                    use_container_width=True
                ):

                    output_df = batch_df.copy()

                    predictions_all = []

                    for name, model in st.session_state.models.items():

                        try:

                            pred = model.predict(batch_X)

                            output_df[name] = [
                                "Parkinson's Disease"
                                if value == 1
                                else "Healthy"
                                for value in pred
                            ]

                            predictions_all.append(pred)

                        except Exception as e:

                            st.warning(
                                f"{name} batch prediction failed: {e}"
                            )

                    if predictions_all:

                        prediction_array = np.array(
                            predictions_all
                        )

                        majority_predictions = []

                        for index in range(
                            prediction_array.shape[1]
                        ):

                            values = prediction_array[:, index]

                            majority_predictions.append(
                                1 if np.sum(values) >
                                len(values) / 2 else 0
                            )

                        output_df["Majority Prediction"] = [
                            "Parkinson's Disease"
                            if value == 1
                            else "Healthy"
                            for value in majority_predictions
                        ]

                        st.session_state.batch_result = output_df

                        st.success(
                            "Batch prediction completed."
                        )

        except Exception as e:

            st.error(
                f"Unable to process batch file: {e}"
            )

    if st.session_state.batch_result is not None:

        st.markdown("---")

        st.subheader("Batch Prediction Results")

        st.dataframe(
            st.session_state.batch_result,
            use_container_width=True,
            hide_index=True
        )

        csv_data = st.session_state.batch_result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Prediction Results",
            data=csv_data,
            file_name="parkinson_batch_predictions.csv",
            mime="text/csv"
        )


# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

elif page == "SHAP Explainability":

    st.subheader("SHAP Explainability")

    st.caption(
        "Explain the contribution of voice features to a model prediction."
    )

    if st.session_state.latest_input is None:

        st.info(
            "Run a prediction from Manual Voice Input before using SHAP."
        )

    else:

        tree_models = [
            name for name in [
                "Random Forest",
                "Extra Trees",
                "Decision Tree",
                "XGBoost"
            ]
            if name in st.session_state.models
        ]

        if not tree_models:

            st.warning(
                "No supported tree-based model is available."
            )

        else:

            selected_model = st.selectbox(
                "Select Explainable Model",
                tree_models
            )

            if st.button(
                "Generate SHAP Explanation",
                use_container_width=True
            ):

                try:

                    import shap

                    model = st.session_state.models[
                        selected_model
                    ]

                    input_df = pd.DataFrame(
                        [st.session_state.latest_input["input"]],
                        columns=feature_columns
                    )

                    pipeline_model = model.named_steps["model"]

                    imputer = model.named_steps["imputer"]

                    transformed_input = imputer.transform(
                        input_df
                    )

                    transformed_background = imputer.transform(
                        X.sample(
                            min(50, len(X)),
                            random_state=42
                        )
                    )

                    explainer = shap.TreeExplainer(
                        pipeline_model,
                        data=transformed_background
                    )

                    shap_values = explainer.shap_values(
                        transformed_input
                    )

                    if isinstance(shap_values, list):

                        if len(shap_values) > 1:
                            values = shap_values[1][0]
                        else:
                            values = shap_values[0][0]

                    else:

                        values = np.array(shap_values)

                        if values.ndim == 3:
                            values = values[0, :, 1]

                        elif values.ndim == 2:
                            values = values[0]

                    shap_df = pd.DataFrame({
                        "Feature": feature_columns,
                        "SHAP Value": values
                    })

                    shap_df["Absolute SHAP"] = (
                        shap_df["SHAP Value"].abs()
                    )

                    shap_df = shap_df.sort_values(
                        "Absolute SHAP",
                        ascending=False
                    ).head(15)

                    st.success(
                        "SHAP explanation generated."
                    )

                    st.subheader("Feature Contribution")

                    fig, ax = plt.subplots(figsize=(9, 6))

                    ax.barh(
                        shap_df["Feature"][::-1],
                        shap_df["SHAP Value"][::-1]
                    )

                    ax.set_xlabel("SHAP Value")
                    ax.set_ylabel("Voice Features")
                    ax.set_title("SHAP Feature Importance")

                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(
                        shap_df.drop(
                            columns=["Absolute SHAP"]
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                except ImportError:

                    st.error(
                        "SHAP is not installed. Run: "
                        "python -m pip install shap"
                    )

                except Exception as e:

                    st.error(
                        f"SHAP explanation could not be generated: {e}"
                    )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.subheader("Model Information")

    model_information = {
        "Logistic Regression":
            "A linear classification algorithm used as a baseline model.",

        "SVM":
            "Support Vector Machine identifies a decision boundary between classes.",

        "KNN":
            "K-Nearest Neighbors predicts a class using nearby training samples.",

        "Decision Tree":
            "A tree-based model that makes predictions through decision rules.",

        "Random Forest":
            "An ensemble of decision trees that combines multiple predictions.",

        "Extra Trees":
            "An ensemble learning method using randomized decision trees.",

        "XGBoost":
            "A gradient boosting algorithm designed for efficient classification.",

        "AdaBoost":
            "An ensemble method that combines multiple weak learners.",

        "Hard Voting":
            "The final class is selected by majority voting among base models.",

        "Soft Voting":
            "The final class is selected using averaged predicted probabilities.",

        "Stacking":
            "A meta-model combines predictions from multiple base models."
    }

    for name, description in model_information.items():

        if name in st.session_state.models:

            with st.expander(name):

                st.write(description)

    st.markdown("---")

    st.subheader("Dataset Information")

    st.write(f"Total Records: {len(df)}")
    st.write(f"Total Features: {len(feature_columns)}")
    st.write(f"Target Column: status")
    st.write("Class 0: Healthy")
    st.write("Class 1: Parkinson's Disease")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 15px 5px;
        color: #94A3B8;
    ">
        <div style="
            color: #38BDF8;
            font-size: 17px;
            font-weight: 600;
        ">
            Parkinson's Disease Voice Classification System
        </div>

        <div style="font-size: 13px; margin-top: 6px;">
            Machine Learning | Ensemble Learning | SHAP Explainability
        </div>

        <div style="font-size: 11px; margin-top: 6px;">
            Research and Educational Purpose Only
        </div>
    </div>
    """,
    unsafe_allow_html=True
)