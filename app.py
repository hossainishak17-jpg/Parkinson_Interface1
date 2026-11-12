
import hashlib
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

try:
    from xgboost import XGBClassifier
except ImportError:
    st.error("XGBoost is missing. Run: pip install xgboost")
    st.stop()

warnings.filterwarnings("ignore")

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Parkinson's Voice Classification",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #24364b;
}

.main .block-container {
    padding-top: 1.2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #174b78 !important;
    font-weight: 700;
}

p, label, li {
    color: #263f59;
}

section[data-testid="stSidebar"] {
    background-color: #edf4fb;
    border-right: 1px solid #c9dced;
}

section[data-testid="stSidebar"] * {
    color: #173d62;
}

[data-testid="stMetric"] {
    background-color: #edf5fd;
    border: 1px solid #c5dced;
    border-left: 4px solid #2878bd;
    border-radius: 9px;
    padding: 13px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #174b78 !important;
}

[data-testid="stFileUploader"] {
    background-color: #f2f7fc;
    border: 1px dashed #82afd5;
    border-radius: 9px;
    padding: 9px;
}

.stButton > button {
    background-color: #246da8;
    color: white;
    border: 1px solid #1d5b91;
    border-radius: 7px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #174b78;
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #d1dfec;
    border-radius: 8px;
}

.stAlert {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# CONSTANTS
# =====================================================

UCI_COLUMNS = [
    "name", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "status",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

IGNORE = {
    "name", "status", "subject_id", "subject",
    "patient_id", "id", "group"
}

MODEL_NAMES = [
    "Logistic Regression",
    "SVM",
    "KNN",
    "Random Forest",
    "XGBoost",
    "Hard Voting",
    "Soft Voting",
    "Stacking"
]


# =====================================================
# SESSION STATE
# =====================================================

DEFAULTS = {
    "main_df": None,
    "train_df": None,
    "X_train": None,
    "y_train": None,
    "groups_train": None,
    "models": {},
    "trained": False,
    "train_hash": None,
    "predictions": None,
    "test_evaluations": {},
    "cv_results": {},
    "cv_details": {},
    "shap_result": None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_models():

    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.groups_train = None
    st.session_state.models = {}
    st.session_state.trained = False
    st.session_state.train_hash = None
    st.session_state.predictions = None
    st.session_state.test_evaluations = {}
    st.session_state.cv_results = {}
    st.session_state.cv_details = {}
    st.session_state.shap_result = None


# =====================================================
# DATASET FUNCTIONS
# =====================================================

def file_hash(file):

    file.seek(0)
    value = hashlib.md5(file.read()).hexdigest()
    file.seek(0)

    return value


def read_data(file):

    file.seek(0)

    if file.name.lower().endswith((".data", ".txt")):
        df = pd.read_csv(file, header=None)
    else:
        df = pd.read_csv(file)

    df.columns = [
        str(c).strip().strip('"').strip("'")
        for c in df.columns
    ]

    normalized = [str(c).strip().lower() for c in df.columns]

    if df.shape[1] == 24 and "status" not in normalized:
        df.columns = UCI_COLUMNS

    return df


def get_status_col(df):

    for col in df.columns:
        if str(col).strip().lower() == "status":
            return col

    return None


def get_groups(df):

    subject_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower() == "subject_id"
        ),
        None
    )

    name_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower() == "name"
        ),
        None
    )

    if subject_col is not None:

        groups = df[subject_col].astype(str).str.strip()

        if groups.isna().any() or groups.isin(["", "nan", "None"]).any():
            return None

        return groups

    if name_col is not None:

        groups = df[name_col].astype(str).str.upper().str.extract(
            r"(S\d+)",
            expand=False
        )

        if groups.isna().any():
            return None

        return groups.astype(str)

    return None


def prepare_data(df, require_target=True):

    target = get_status_col(df)

    if require_target and target is None:
        raise ValueError("Dataset must contain a status column.")

    y = None

    if target is not None:

        y = pd.to_numeric(df[target], errors="coerce")

        if y.isna().any():
            raise ValueError("Invalid or missing status values.")

        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("Status must contain only 0 and 1.")

        if require_target and set(y.unique()) != {0, 1}:
            raise ValueError("Training data must contain both classes.")

        y = y.astype(int)

    drop_cols = [
        c for c in df.columns
        if str(c).strip().lower() in IGNORE
    ]

    X = df.drop(
        columns=drop_cols,
        errors="ignore"
    ).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No numeric features found.")

    if X.isna().all(axis=1).any():
        raise ValueError("Some rows have no usable numeric features.")

    groups = get_groups(df)

    return X, y, groups


def align_features(X, features):

    X = X.copy()

    for col in features:
        if col not in X.columns:
            X[col] = np.nan

    X = X.reindex(columns=features)

    return X


# =====================================================
# MODEL PIPELINES
# =====================================================

def make_scaled(model):

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def build_models(y):

    class_counts = pd.Series(y).value_counts()

    if len(class_counts) != 2:
        raise ValueError("Both classes are required for training.")

    min_class = int(class_counts.min())

    if min_class < 2:
        raise ValueError(
            "At least two samples from each class are required "
            "for Stacking."
        )

    inner_cv = min(3, min_class)

    base = [
        (
            "lr",
            make_scaled(
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ),
        (
            "svm",
            make_scaled(
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=42
                )
            )
        ),
        (
            "rf",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ])
        )
    ]

    return {

        "Logistic Regression": make_scaled(
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        ),

        "SVM": make_scaled(
            SVC(
                kernel="rbf",
                C=1,
                gamma="scale",
                probability=True,
                random_state=42
            )
        ),

        "KNN": make_scaled(
            KNeighborsClassifier(
                n_neighbors=min(5, len(y)),
                metric="minkowski"
            )
        ),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42,
                    eval_metric="logloss",
                    objective="binary:logistic",
                    n_jobs=1,
                    verbosity=0
                )
            )
        ]),

        "Hard Voting": VotingClassifier(
            estimators=base,
            voting="hard"
        ),

        "Soft Voting": VotingClassifier(
            estimators=base,
            voting="soft"
        ),

        "Stacking": StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(
                max_iter=2000,
                random_state=42
            ),
            cv=StratifiedKFold(
                n_splits=inner_cv,
                shuffle=True,
                random_state=42
            ),
            stack_method="predict_proba"
        )
    }


def train_all(X, y):

    fitted = {}
    models = build_models(y)

    for name, model in models.items():

        model.fit(X, y)
        fitted[name] = model

    return fitted


# =====================================================
# PREDICTION SCORES
# =====================================================

def positive_probability(model, X):

    try:

        probabilities = model.predict_proba(X)
        classes = list(model.classes_)

        if 1 not in classes:
            return np.zeros(len(X))

        return probabilities[:, classes.index(1)]

    except Exception:
        return None


def hard_vote_score(model, X):

    votes = []

    for fitted_model in model.estimators_:

        votes.append(
            fitted_model.predict(X).astype(float)
        )

    return np.mean(votes, axis=0)


def score_vector(model, X, name):

    if name == "Hard Voting":
        return hard_vote_score(model, X)

    return positive_probability(model, X)


def confidence_vector(pred, score):

    if score is None:
        return np.full(len(pred), np.nan)

    return np.where(
        np.asarray(pred) == 1,
        score,
        1 - score
    )


# =====================================================
# EVALUATION METRICS
# =====================================================

def metric_dict(y, pred, score=None):

    y = np.asarray(y)
    pred = np.asarray(pred)

    auc = np.nan

    if score is not None and len(np.unique(y)) == 2:

        try:
            auc = roc_auc_score(y, score)
        except Exception:
            pass

    return {

        "Accuracy": accuracy_score(y, pred),

        "Precision": precision_score(
            y, pred, zero_division=0
        ),

        "Recall": recall_score(
            y, pred, zero_division=0
        ),

        "F1-Score": f1_score(
            y, pred, zero_division=0
        ),

        "ROC-AUC": auc,

        "MCC": matthews_corrcoef(y, pred)
    }


# =====================================================
# RECORDING-LEVEL EVALUATION
# =====================================================

def run_random_split(X, y):

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = train_all(Xtr, ytr)

    rows = []
    details = {}

    for name, model in models.items():

        pred = model.predict(Xte).astype(int)
        score = score_vector(model, Xte, name)

        rows.append({
            "Model": name,
            **metric_dict(yte, pred, score)
        })

        details[name] = {
            "y_true": yte.to_numpy(),
            "y_pred": pred,
            "score": score
        }

    return pd.DataFrame(rows), details


# =====================================================
# SUBJECT-DISJOINT GROUP K-FOLD
# =====================================================

def run_subject_cv(X, y, groups):

    if groups is None:
        raise ValueError(
            "Subject IDs not found. Provide subject_id or "
            "name values containing S-number."
        )

    if groups.nunique() < 5:
        raise ValueError(
            "At least 5 unique subjects are required "
            "for 5-fold GroupKFold."
        )

    cv = GroupKFold(n_splits=5)

    oof = {

        name: {
            "pred": np.full(len(y), -1, dtype=int),
            "score": np.full(len(y), np.nan)
        }

        for name in MODEL_NAMES
    }

    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        Xtr = X.iloc[train_idx]
        Xte = X.iloc[test_idx]

        ytr = y.iloc[train_idx]
        yte = y.iloc[test_idx]

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(
                f"Subject overlap found in fold {fold}."
            )

        if ytr.nunique() != 2:
            raise ValueError(
                f"Fold {fold}: training data does not contain both classes."
            )

        fitted = train_all(Xtr, ytr)

        for name, model in fitted.items():

            pred = model.predict(Xte).astype(int)
            score = score_vector(model, Xte, name)

            oof[name]["pred"][test_idx] = pred

            if score is not None:
                oof[name]["score"][test_idx] = score

            fold_rows.append({

                "Fold": fold,
                "Model": name,
                "Train Records": len(train_idx),
                "Test Records": len(test_idx),
                "Train Subjects": len(train_groups),
                "Test Subjects": len(test_groups),

                **metric_dict(yte, pred, score)
            })

    fold_df = pd.DataFrame(fold_rows)

    summary = []
    details = {}

    for name in MODEL_NAMES:

        pred = oof[name]["pred"]
        valid = pred >= 0

        score = oof[name]["score"]

        if np.isfinite(score[valid]).all():
            score_arg = score[valid]
        else:
            score_arg = None

        y_true = y.to_numpy()[valid]
        y_pred = pred[valid]

        summary.append({

            "Model": name,

            **metric_dict(
                y_true,
                y_pred,
                score_arg
            )
        })

        details[name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "score": score_arg
        }

    return (
        pd.DataFrame(summary),
        fold_df,
        details
    )


# =====================================================
# DISPLAY HELPERS
# =====================================================

def show_metrics(df):

    numeric_cols = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "MCC"
    ]

    available = [
        c for c in numeric_cols if c in df.columns
    ]

    st.dataframe(
        df.style.format(
            "{:.4f}",
            subset=available,
            na_rep="N/A"
        ),
        use_container_width=True
    )


def download_csv(df, filename, key):

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        filename,
        "text/csv",
        key=key
    )


def show_confusion_matrix(detail, title):

    cm = confusion_matrix(
        detail["y_true"],
        detail["y_pred"],
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(
        [0, 1],
        ["Healthy", "Parkinson's"]
    )

    ax.set_yticks(
        [0, 1],
        ["Healthy", "Parkinson's"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for r in range(2):
        for c in range(2):

            ax.text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
                color="black"
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def show_roc_curve(detail):

    score = detail["score"]
    y_true = detail["y_true"]

    if score is None or len(np.unique(y_true)) != 2:
        st.info("ROC curve is not available for this result.")
        return

    fpr, tpr, _ = roc_curve(y_true, score)

    auc = roc_auc_score(y_true, score)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Parkinson's Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Model Training",
        "Evaluation",
        "Prediction",
        "Model Comparison",
        "SHAP Explainability"
    ]
)


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("Parkinson's Disease Voice Classification")

    st.write(
        "Machine Learning-Based Parkinson's Disease "
        "Voice Classification"
    )

    file = st.file_uploader(
        "Main Dataset",
        type=["csv", "data", "txt"],
        key="main_upload"
    )

    if file:

        try:

            st.session_state.main_df = read_data(file)
            st.success("Main dataset loaded successfully.")

        except Exception as e:
            st.error(str(e))

    if st.session_state.main_df is not None:

        try:

            df = st.session_state.main_df

            X, y, groups = prepare_data(df)

            a, b, c, d = st.columns(4)

            a.metric("Records", len(df))
            b.metric("Features", X.shape[1])

            c.metric(
                "Subjects",
                groups.nunique() if groups is not None else "Not found"
            )

            d.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.subheader("Class Distribution")

            distribution = y.value_counts().sort_index()

            distribution.index = [
                "Healthy" if i == 0 else "Parkinson's"
                for i in distribution.index
            ]

            st.bar_chart(distribution)

            if groups is None:

                st.warning(
                    "Subject IDs not found. Subject-disjoint "
                    "evaluation requires subject identifiers."
                )

        except Exception as e:
            st.error(f"Dataset error: {e}")

    else:
        st.info("Upload the main Parkinson's dataset.")


# =====================================================
# MODEL TRAINING
# =====================================================

elif page == "Model Training":

    st.title("Model Training")

    file = st.file_uploader(
        "Training Dataset",
        type=["csv", "data", "txt"],
        key="train_upload"
    )

    if file:

        try:

            current_hash = file_hash(file)

            if current_hash != st.session_state.train_hash:

                reset_models()

                df = read_data(file)

                X, y, groups = prepare_data(df)

                st.session_state.train_df = df
                st.session_state.X_train = X
                st.session_state.y_train = y
                st.session_state.groups_train = groups
                st.session_state.train_hash = current_hash

            X = st.session_state.X_train
            y = st.session_state.y_train

            a, b, c = st.columns(3)

            a.metric("Training Records", len(X))
            b.metric("Features", X.shape[1])
            c.metric("Models", 8)

            st.subheader("Training Dataset Preview")

            st.dataframe(
                st.session_state.train_df.head(8),
                use_container_width=True
            )

            if st.button(
                "Train All 8 Models",
                type="primary"
            ):

                with st.spinner("Training all models..."):

                    st.session_state.models = train_all(X, y)

                    st.session_state.trained = True

                st.success("All 8 models trained successfully.")

            if st.session_state.trained:

                st.subheader("Training Status")

                status_df = pd.DataFrame({
                    "Model": list(st.session_state.models.keys()),
                    "Status": ["Trained"] * len(st.session_state.models)
                })

                st.dataframe(
                    status_df,
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Training error: {e}")

    else:

        st.info("Upload the labelled training dataset.")


# =====================================================
# EVALUATION
# =====================================================

elif page == "Evaluation":

    st.title("Model Evaluation")

    file = st.file_uploader(
        "Evaluation Dataset",
        type=["csv", "data", "txt"],
        key="evaluation_upload"
    )

    if file:

        try:

            df = read_data(file)

            X, y, groups = prepare_data(df)

            mode = st.radio(
                "Evaluation Method",
                [
                    "Recording-Level 80/20 Split",
                    "Subject-Disjoint 5-Fold GroupKFold"
                ],
                horizontal=True
            )

            if st.button(
                "Run Evaluation",
                type="primary"
            ):

                if mode.startswith("Recording"):

                    with st.spinner(
                        "Running recording-level evaluation..."
                    ):

                        summary, details = run_random_split(X, y)

                    st.session_state.cv_results[
                        "Recording-Level"
                    ] = summary

                    st.session_state.cv_details[
                        "Recording-Level"
                    ] = details

                else:

                    with st.spinner(
                        "Running all five GroupKFold folds..."
                    ):

                        summary, folds, details = run_subject_cv(
                            X, y, groups
                        )

                    st.session_state.cv_results[
                        "Subject-Disjoint"
                    ] = summary

                    st.session_state.cv_results[
                        "Fold Details"
                    ] = folds

                    st.session_state.cv_details[
                        "Subject-Disjoint"
                    ] = details

                st.success("Evaluation completed successfully.")

            key = (
                "Recording-Level"
                if mode.startswith("Recording")
                else "Subject-Disjoint"
            )

            if key in st.session_state.cv_results:

                results = st.session_state.cv_results[key]

                st.subheader(key + " Results")

                show_metrics(results)

                download_csv(
                    results,
                    "evaluation_results.csv",
                    "evaluation_download"
                )

                if (
                    key == "Subject-Disjoint"
                    and "Fold Details" in st.session_state.cv_results
                ):

                    st.subheader("Individual Fold Results")

                    folds = st.session_state.cv_results["Fold Details"]

                    st.dataframe(
                        folds,
                        use_container_width=True
                    )

                    download_csv(
                        folds,
                        "five_fold_results.csv",
                        "fold_download"
                    )

                st.subheader("Accuracy Comparison")

                st.bar_chart(
                    results.set_index("Model")[["Accuracy"]]
                )

                selected = st.selectbox(
                    "Select Model",
                    MODEL_NAMES,
                    key="evaluation_model"
                )

                detail = st.session_state.cv_details[key][selected]

                st.subheader("Confusion Matrix")

                show_confusion_matrix(
                    detail,
                    selected + " Confusion Matrix"
                )

                st.subheader("ROC Curve")

                show_roc_curve(detail)

                st.subheader("Classification Report")

                report = classification_report(
                    detail["y_true"],
                    detail["y_pred"],
                    labels=[0, 1],
                    target_names=["Healthy", "Parkinson's"],
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(
                    pd.DataFrame(report).transpose(),
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Evaluation error: {e}")

    else:

        st.info("Upload a labelled evaluation dataset.")


# =====================================================
# PREDICTION
# =====================================================

elif page == "Prediction":

    st.title("Prediction")

    if not st.session_state.trained:

        st.warning("Train all models first.")
        st.stop()

    files = st.file_uploader(
        "Testing Dataset(s)",
        type=["csv", "data", "txt"],
        accept_multiple_files=True,
        key="test_upload"
    )

    if files and st.button(
        "Run Predictions",
        type="primary"
    ):

        outputs = []
        evaluations = {}

        for file in files:

            try:

                df = read_data(file)

                Xtest, ytest, _ = prepare_data(
                    df,
                    require_target=False
                )

                features = list(
                    st.session_state.X_train.columns
                )

                Xtest = align_features(Xtest, features)

                predictions = {}
                scores = {}

                for name, model in st.session_state.models.items():

                    pred = model.predict(Xtest).astype(int)

                    score = score_vector(
                        model,
                        Xtest,
                        name
                    )

                    predictions[name] = pred
                    scores[name] = score

                    confidence = confidence_vector(
                        pred,
                        score
                    )

                    for i in range(len(pred)):

                        outputs.append({

                            "File": file.name,
                            "Record": i + 1,
                            "Model": name,

                            "Prediction": (
                                "Parkinson's"
                                if pred[i] == 1
                                else "Healthy"
                            ),

                            "Parkinson's Score (%)": (
                                float(score[i] * 100)
                                if score is not None
                                else np.nan
                            ),

                            "Confidence (%)": (
                                float(confidence[i] * 100)
                                if not np.isnan(confidence[i])
                                else np.nan
                            )
                        })

                table = pd.DataFrame({
                    "Record": np.arange(1, len(Xtest) + 1)
                })

                for name, pred in predictions.items():

                    table[name] = [
                        "Parkinson's" if v == 1 else "Healthy"
                        for v in pred
                    ]

                table["Agreement (%)"] = [

                    max(
                        pd.Series([
                            predictions[n][i]
                            for n in MODEL_NAMES
                        ]).value_counts()
                    ) / len(MODEL_NAMES) * 100

                    for i in range(len(Xtest))
                ]

                if ytest is not None:

                    table["Actual Status"] = [
                        "Parkinson's" if v == 1 else "Healthy"
                        for v in ytest
                    ]

                st.subheader(file.name)

                st.dataframe(
                    table,
                    use_container_width=True
                )

                if ytest is not None:

                    rows = []

                    for name, model in st.session_state.models.items():

                        rows.append({

                            "Model": name,

                            **metric_dict(
                                ytest,
                                model.predict(Xtest),
                                scores[name]
                            )
                        })

                    evaluations[file.name] = pd.DataFrame(rows)

                    st.subheader("Testing Metrics")

                    show_metrics(evaluations[file.name])

            except Exception as e:

                st.error(f"{file.name}: {e}")

        if outputs:

            st.session_state.predictions = pd.DataFrame(outputs)

            st.session_state.test_evaluations = evaluations

            st.subheader("All Model Predictions")

            st.dataframe(
                st.session_state.predictions,
                use_container_width=True
            )

            download_csv(
                st.session_state.predictions,
                "all_model_predictions.csv",
                "prediction_download"
            )


# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    options = [
        k for k in [
            "Recording-Level",
            "Subject-Disjoint"
        ]
        if k in st.session_state.cv_results
    ]

    if options:

        selected = st.selectbox(
            "Evaluation Method",
            options
        )

        results = st.session_state.cv_results[selected]

        st.subheader(selected + " Comparison")

        show_metrics(results)

        chart_cols = [
            c for c in [
                "Accuracy",
                "F1-Score",
                "ROC-AUC"
            ]
            if c in results.columns
        ]

        st.bar_chart(
            results.set_index("Model")[chart_cols]
        )

    if st.session_state.test_evaluations:

        st.subheader("Testing Dataset Comparison")

        filename = st.selectbox(
            "Testing File",
            list(st.session_state.test_evaluations)
        )

        show_metrics(
            st.session_state.test_evaluations[filename]
        )

    if not options and not st.session_state.test_evaluations:

        st.info("Run an evaluation or prediction first.")


# =====================================================
# SHAP EXPLAINABILITY
# =====================================================

elif page == "SHAP Explainability":

    st.title("SHAP Explainability")

    if not st.session_state.trained:

        st.warning("Train all models first.")
        st.stop()

    try:
        import shap

    except ImportError:

        st.error("Install SHAP using: pip install shap")
        st.stop()

    name = st.selectbox(
        "Select Model",
        MODEL_NAMES,
        key="shap_model"
    )

    file = st.file_uploader(
        "Testing Dataset for SHAP",
        type=["csv", "data", "txt"],
        key="shap_test"
    )

    if file and st.button(
        "Generate SHAP Explanation",
        type="primary"
    ):

        try:

            df = read_data(file)

            Xtest, _, _ = prepare_data(
                df,
                require_target=False
            )

            features = list(
                st.session_state.X_train.columns
            )

            Xtest = align_features(
                Xtest,
                features
            ).head(5)

            background = st.session_state.X_train.sample(
                min(12, len(st.session_state.X_train)),
                random_state=42
            )

            model = st.session_state.models[name]

            def predict_fn(values):

                frame = pd.DataFrame(
                    values,
                    columns=features
                )

                score = score_vector(
                    model,
                    frame,
                    name
                )

                if score is not None:
                    return score

                return model.predict(frame).astype(float)

            with st.spinner("Calculating SHAP values..."):

                explainer = shap.KernelExplainer(
                    predict_fn,
                    background
                )

                values = explainer.shap_values(
                    Xtest,
                    nsamples=60
                )

            if isinstance(values, list):
                values = values[-1]

            values = np.asarray(values)

            if values.ndim == 3:
                values = values[:, :, -1]

            values = np.squeeze(values)

            if values.ndim == 1:
                values = values.reshape(1, -1)

            if values.shape != Xtest.shape:

                raise ValueError(
                    "Unexpected SHAP output shape."
                )

            importance = pd.DataFrame({

                "Feature": features,

                "Mean Absolute SHAP": np.abs(
                    values
                ).mean(axis=0)

            }).sort_values(
                "Mean Absolute SHAP",
                ascending=False
            )

            st.session_state.shap_result = {

                "model": name,
                "importance": importance,
                "values": values,
                "data": Xtest
            }

            st.success("SHAP explanation generated successfully.")

        except Exception as e:

            st.error(f"SHAP error: {e}")

    result = st.session_state.shap_result

    if result and result["model"] == name:

        st.subheader("Feature Importance")

        st.dataframe(
            result["importance"],
            use_container_width=True
        )

        top = result["importance"].head(15).iloc[::-1]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            top["Feature"],
            top["Mean Absolute SHAP"]
        )

        ax.set_xlabel("Mean Absolute SHAP Value")
        ax.set_title("Top 15 Important Features")

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("SHAP Summary Plot")

        try:

            shap.summary_plot(
                result["values"],
                result["data"],
                feature_names=list(
                    result["data"].columns
                ),
                show=False
            )

            fig = plt.gcf()

            st.pyplot(
                fig,
                clear_figure=True
            )

            plt.close(fig)

        except Exception as e:

            st.warning(
                f"SHAP summary plot unavailable: {e}"
            )

        download_csv(
            result["importance"],
            "shap_feature_importance.csv",
            "shap_download"
        )
