import os
import warnings
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    StratifiedKFold
)
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
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Parkinson's Voice Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROFESSIONAL NAVY BLUE THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #071426 0%,
        #101D35 55%,
        #172943 100%
    );
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background: #091528;
    border-right: 1px solid #263954;
}

h1, h2, h3, h4 {
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
    padding: 10px 18px;
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

div[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

.stTextInput input,
.stNumberInput input {
    background-color: #14243D;
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: #14243D;
}

.stDownloadButton > button {
    background: #075985;
    color: white;
    border-radius: 8px;
}

hr {
    border-color: #334155;
}

[data-testid="stDataFrame"] {
    border: 1px solid #29415F;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CONSTANTS
# =========================================================

MODEL_NAMES = [
    "Logistic Regression",
    "SVM",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "Hard Voting",
    "Soft Voting",
    "Stacking"
]

CLASS_NAMES = {
    0: "Healthy",
    1: "Parkinson's"
}

METRIC_COLUMNS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "MCC"
]


# =========================================================
# 4. SESSION STATE
# =========================================================

DEFAULTS = {
    "models": {},
    "metrics": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "trained": False,
    "dataset_signature": None,
    "latest_input": None,
    "latest_results": None,
    "latest_source": None,
    "latest_shap": None,
    "subject_metrics": None,
    "subject_folds": None,
    "subject_info": None
}


def initialize_session():

    for key, value in DEFAULTS.items():

        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():

    for key, value in DEFAULTS.items():
        st.session_state[key] = value


initialize_session()


# =========================================================
# 5. DATASET LOADING
# =========================================================

def load_dataset(uploaded_file):

    if uploaded_file is not None:

        uploaded_file.seek(0)

        return pd.read_csv(uploaded_file)

    folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    possible_files = [
        "parkinsons.csv",
        "parkinsons.data"
    ]

    for filename in possible_files:

        path = os.path.join(
            folder,
            filename
        )

        if os.path.isfile(path):

            return pd.read_csv(path)

    return None


# =========================================================
# 6. DATASET PREPARATION
# =========================================================

def prepare_dataset(data):

    df = data.copy()

    df.columns = (
        df.columns.astype(str).str.strip()
    )

    # Remove duplicate column names

    if df.columns.duplicated().any():

        raise ValueError(
            "Dataset contains duplicate column names."
        )

    # Target column

    if "status" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'status' column."
        )

    # Extract subject ID

    if "subject_id" not in df.columns:

        if "name" not in df.columns:

            raise ValueError(
                "Dataset must contain 'name' or 'subject_id'."
            )

        extracted = (
            df["name"]
            .astype(str)
            .str.extract(
                r"(S\d+)",
                expand=False
            )
        )

        df["subject_id"] = extracted

    # Clean subject ID

    df["subject_id"] = (
        df["subject_id"]
        .astype("string")
        .str.strip()
    )

    df["subject_id"] = df["subject_id"].replace(
        "",
        pd.NA
    )

    # Target conversion

    y = pd.to_numeric(
        df["status"],
        errors="coerce"
    )

    valid = y.isin([0, 1])

    df = df.loc[valid].copy().reset_index(drop=True)

    y = (
        y.loc[valid]
        .astype(int)
        .reset_index(drop=True)
    )

    # Check subjects

    if df["subject_id"].isna().any():

        raise ValueError(
            "Some subject IDs could not be extracted. "
            "Check the name column or provide subject_id."
        )

    # Numerical voice features

    excluded = [
        "name",
        "status",
        "subject_id"
    ]

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.notna().any():

            df[column] = converted

            features.append(column)

    if len(features) == 0:

        raise ValueError(
            "No numerical voice features found."
        )

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Completely empty columns are not usable

    empty_columns = X.columns[
        X.isna().all()
    ].tolist()

    if empty_columns:

        raise ValueError(
            "Features contain no usable numeric values: "
            + ", ".join(empty_columns)
        )

    if X.shape[0] < 10:

        raise ValueError(
            "Dataset contains too few valid records."
        )

    if y.nunique() != 2:

        raise ValueError(
            "Dataset must contain both classes: "
            "Healthy (0) and Parkinson's (1)."
        )

    if df["subject_id"].nunique() < 5:

        raise ValueError(
            "At least five unique subjects are required "
            "for 5-fold GroupKFold evaluation."
        )

    return df, X, y, features


# =========================================================
# 7. DATASET SIGNATURE
# =========================================================

def get_dataset_signature(df):

    hashed = pd.util.hash_pandas_object(
        df,
        index=True
    ).values.tobytes()

    digest = hashlib.sha256(
        hashed
    ).hexdigest()

    return digest


# =========================================================
# 8. BASE MODELS
# =========================================================

def create_base_models(training_size):

    k = max(
        1,
        min(5, training_size)
    )

    models = {

        "Logistic Regression": Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            ("scaler", StandardScaler()),

            ("model", LogisticRegression(
                max_iter=2000,
                random_state=42
            ))

        ]),

        "SVM": Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            ("scaler", StandardScaler()),

            ("model", SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            ))

        ]),

        "KNN": Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            ("scaler", StandardScaler()),

            ("model", KNeighborsClassifier(
                n_neighbors=k
            ))

        ]),

        "Decision Tree": Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            ("model", DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            ))

        ]),

        "Random Forest": Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
                class_weight="balanced"
            ))

        ])

    }

    return models


# =========================================================
# 9. ALL EIGHT MODELS
# =========================================================

def create_all_models(
    training_size,
    minority_count
):

    if minority_count < 2:

        raise ValueError(
            "Not enough minority-class samples "
            "for ensemble training."
        )

    base_models = create_base_models(
        training_size
    )

    estimators = [

        (
            name.replace(" ", "_"),
            clone(model)
        )

        for name, model in base_models.items()

    ]

    cv_folds = min(
        3,
        minority_count
    )

    stacking_cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42
    )

    all_models = dict(base_models)

    all_models["Hard Voting"] = VotingClassifier(

        estimators=estimators,

        voting="hard",

        n_jobs=1

    )

    all_models["Soft Voting"] = VotingClassifier(

        estimators=estimators,

        voting="soft",

        n_jobs=1

    )

    all_models["Stacking"] = StackingClassifier(

        estimators=estimators,

        final_estimator=LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        cv=stacking_cv,

        n_jobs=1

    )

    return all_models


# =========================================================
# 10. CLASS PROBABILITIES
# =========================================================

def get_model_probabilities(
    model,
    X_input,
    model_name
):

    # Hard voting has no native predict_proba.

    if model_name == "Hard Voting":

        probabilities = []

        for estimator in model.estimators_:

            if not hasattr(
                estimator,
                "predict_proba"
            ):
                continue

            prob = estimator.predict_proba(
                X_input
            )[0]

            classes = list(
                estimator.classes_
            )

            if 0 in classes and 1 in classes:

                probabilities.append([

                    prob[classes.index(0)],

                    prob[classes.index(1)]

                ])

        if not probabilities:

            return None, "Unavailable"

        average = np.mean(
            probabilities,
            axis=0
        )

        return (
            average,
            "Estimated average base-model probability"
        )

    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(
            X_input
        )[0]

        classes = list(
            model.classes_
        )

        if 0 in classes and 1 in classes:

            result = np.array([

                prob[classes.index(0)],

                prob[classes.index(1)]

            ])

            return result, "Model probability"

    return None, "Unavailable"


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

def compare_models(
    input_df,
    models
):

    rows = []

    predictions = []

    for name in MODEL_NAMES:

        model = models[name]

        prediction = int(
            model.predict(input_df)[0]
        )

        probabilities, confidence_type = (
            get_model_probabilities(
                model,
                input_df,
                name
            )
        )

        if probabilities is not None:

            healthy_confidence = float(
                probabilities[0]
            )

            pd_confidence = float(
                probabilities[1]
            )

            predicted_confidence = float(
                probabilities[prediction]
            )

        else:

            healthy_confidence = np.nan
            pd_confidence = np.nan
            predicted_confidence = np.nan

        predictions.append(prediction)

        rows.append({

            "Model": name,

            "Predicted Class": CLASS_NAMES[
                prediction
            ],

            "Healthy Confidence (%)": (
                healthy_confidence * 100
                if not np.isnan(healthy_confidence)
                else np.nan
            ),

            "Parkinson's Confidence (%)": (
                pd_confidence * 100
                if not np.isnan(pd_confidence)
                else np.nan
            ),

            "Predicted Class Confidence (%)": (
                predicted_confidence * 100
                if not np.isnan(predicted_confidence)
                else np.nan
            ),

            "Confidence Type": confidence_type

        })

    result_df = pd.DataFrame(rows)

    pd_count = predictions.count(1)

    healthy_count = predictions.count(0)

    majority_class = (
        1 if pd_count > healthy_count else 0
    )

    agreement = (
        max(pd_count, healthy_count)
        / len(predictions)
    ) * 100

    return (
        result_df,
        predictions,
        majority_class,
        agreement,
        pd_count,
        healthy_count
    )


# =========================================================
# 12. MODEL EVALUATION
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    result = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test,
            predictions
        ),

        "ROC-AUC": np.nan

    }

    # ROC-AUC requires both classes in test data

    if y_test.nunique() == 2:

        try:

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    X_test
                )

                classes = list(
                    model.classes_
                )

                if 1 in classes:

                    positive_index = classes.index(1)

                    result["ROC-AUC"] = roc_auc_score(
                        y_test,
                        probabilities[:, positive_index]
                    )

        except Exception:

            result["ROC-AUC"] = np.nan

    return result


# =========================================================
# 13. RANDOM SPLIT TRAINING
# =========================================================

def train_all_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    if y_train.nunique() != 2:

        raise ValueError(
            "Training split does not contain both classes."
        )

    if y_test.nunique() != 2:

        raise ValueError(
            "Testing split does not contain both classes."
        )

    models = create_all_models(

        len(X_train),

        int(y_train.value_counts().min())

    )

    trained_models = {}

    result_rows = []

    progress = st.progress(0)

    status_text = st.empty()

    for i, (name, model) in enumerate(
        models.items()
    ):

        status_text.write(
            f"Training {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        result = evaluate_model(
            model,
            X_test,
            y_test
        )

        result["Model"] = name

        result_rows.append(result)

        progress.progress(
            (i + 1) / len(models)
        )

    progress.empty()

    status_text.empty()

    return (

        trained_models,

        pd.DataFrame(result_rows),

        X_train,

        X_test,

        y_train,

        y_test

    )


# =========================================================
# 14. SHAP EXPLANATION
# =========================================================

def generate_shap_explanation(
    input_df,
    feature_names,
    random_forest_pipeline
):

    import shap

    imputer = random_forest_pipeline.named_steps[
        "imputer"
    ]

    rf_model = random_forest_pipeline.named_steps[
        "model"
    ]

    transformed = imputer.transform(
        input_df
    )

    explainer = shap.TreeExplainer(
        rf_model
    )

    values = explainer.shap_values(
        transformed
    )

    classes = list(
        rf_model.classes_
    )

    positive_index = (
        classes.index(1)
        if 1 in classes
        else 0
    )

    # SHAP output compatibility:
    # Older SHAP: list of class arrays
    # Newer SHAP: 3D ndarray
    # Some versions: 2D ndarray

    if isinstance(values, list):

        if positive_index < len(values):

            shap_values = np.asarray(
                values[positive_index]
            )[0]

        else:

            raise ValueError(
                "Positive class SHAP values unavailable."
            )

    else:

        values = np.asarray(values)

        if values.ndim == 3:

            if values.shape[-1] == len(classes):

                shap_values = values[
                    0,
                    :,
                    positive_index
                ]

            elif values.shape[0] == len(classes):

                shap_values = values[
                    positive_index,
                    0,
                    :
                ]

            elif values.shape[1] == len(classes):

                shap_values = values[
                    0,
                    positive_index,
                    :
                ]

            else:

                raise ValueError(
                    f"Unsupported SHAP shape: {values.shape}"
                )

        elif values.ndim == 2:

            shap_values = values[0]

        else:

            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

    shap_values = np.asarray(
        shap_values
    ).reshape(-1)

    if len(shap_values) != len(feature_names):

        raise ValueError(
            "SHAP feature dimension does not match "
            "the original input features."
        )

    shap_df = pd.DataFrame({

        "Feature": feature_names,

        "Input Value": [
            input_df.iloc[0][feature]
            for feature in feature_names
        ],

        "SHAP Value": shap_values,

        "Absolute Impact": np.abs(
            shap_values
        )

    })

    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    ).reset_index(drop=True)

    return shap_df


# =========================================================
# 15. SUBJECT-DISJOINT FIVE-FOLD EVALUATION
# =========================================================

def run_subject_disjoint_evaluation(
    X,
    y,
    groups
):

    gkf = GroupKFold(
        n_splits=5
    )

    fold_results = []

    progress = st.progress(0)

    status_text = st.empty()

    splits = list(
        gkf.split(X, y, groups)
    )

    for fold_number, (
        train_idx,
        test_idx
    ) in enumerate(splits, start=1):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        train_groups = set(
            groups.iloc[train_idx]
        )

        test_groups = set(
            groups.iloc[test_idx]
        )

        overlap = (
            train_groups.intersection(
                test_groups
            )
        )

        if overlap:

            raise ValueError(
                f"Subject leakage detected in Fold {fold_number}."
            )

        if y_train.nunique() != 2:

            raise ValueError(
                f"Fold {fold_number}: training data "
                "does not contain both classes."
            )

        if y_train.value_counts().min() < 2:

            raise ValueError(
                f"Fold {fold_number}: insufficient samples "
                "for ensemble stacking."
            )

        status_text.write(
            f"Evaluating Fold {fold_number}/5..."
        )

        models = create_all_models(

            len(X_train),

            int(y_train.value_counts().min())

        )

        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            result = evaluate_model(
                model,
                X_test,
                y_test
            )

            result.update({

                "Fold": fold_number,

                "Model": name,

                "Train Records": len(train_idx),

                "Test Records": len(test_idx),

                "Train Subjects": len(train_groups),

                "Test Subjects": len(test_groups),

                "Subject Overlap": len(overlap)

            })

            fold_results.append(result)

        progress.progress(
            fold_number / 5
        )

    progress.empty()

    status_text.empty()

    fold_df = pd.DataFrame(
        fold_results
    )

    # Mean and standard deviation across five folds

    summary_rows = []

    for name in MODEL_NAMES:

        model_data = fold_df[
            fold_df["Model"] == name
        ]

        row = {
            "Model": name
        }

        for metric in METRIC_COLUMNS:

            row[metric + " Mean"] = (
                model_data[metric].mean()
            )

            row[metric + " Std"] = (
                model_data[metric].std(ddof=1)
                if model_data[metric].notna().sum() > 1
                else np.nan
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    return fold_df, summary_df


# =========================================================
# 16. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title(
    "🧠 Parkinson's AI System"
)

st.sidebar.caption(
    "Voice Classification Platform"
)

uploaded_dataset = st.sidebar.file_uploader(
    "Upload Training Dataset",
    type=["csv", "data"],
    key="training_dataset"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [
        "Dashboard",
        "Prediction",
        "All Models Comparison",
        "SHAP Explainability",
        "Model Evaluation",
        "Subject-Disjoint Evaluation",
        "Model Information"
    ]

)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Machine Learning | Ensemble Learning | SHAP"
)


# =========================================================
# 17. LOAD DATASET
# =========================================================

try:

    raw_data = load_dataset(
        uploaded_dataset
    )

    if raw_data is None:

        st.title(
            "Parkinson's Disease Voice Classification"
        )

        st.warning(
            "Upload your dataset or place "
            "'parkinsons.csv' in the same folder "
            "as app.py."
        )

        st.stop()

    df, X, y, feature_columns = prepare_dataset(
        raw_data
    )

except Exception as error:

    st.error(
        f"Dataset error: {error}"
    )

    st.stop()


# =========================================================
# 18. DATASET CHANGE DETECTION
# =========================================================

signature = get_dataset_signature(
    df
)

if (
    st.session_state.dataset_signature
    != signature
):

    reset_session()

    st.session_state.dataset_signature = signature


# =========================================================
# 19. TRAIN MODELS
# =========================================================

if not st.session_state.trained:

    with st.spinner(
        "Training all eight machine learning models..."
    ):

        try:

            (

                trained_models,

                metrics,

                X_train,

                X_test,

                y_train,

                y_test

            ) = train_all_models(
                X,
                y
            )

            st.session_state.models = trained_models

            st.session_state.metrics = metrics

            st.session_state.X_train = X_train

            st.session_state.X_test = X_test

            st.session_state.y_train = y_train

            st.session_state.y_test = y_test

            st.session_state.trained = True

        except Exception as error:

            st.error(
                f"Model training failed: {error}"
            )

            st.stop()


# =========================================================
# 20. APPLICATION HEADER
# =========================================================

st.title(
    "Parkinson's Disease Voice Classification"
)

st.caption(
    "Machine Learning | Ensemble Learning | Explainable AI"
)

st.markdown("---")


# =========================================================
# 21. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader(
        "System Dashboard"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        len(df)
    )

    c2.metric(
        "Voice Features",
        len(feature_columns)
    )

    c3.metric(
        "Healthy Records",
        int((y == 0).sum())
    )

    c4.metric(
        "Parkinson's Records",
        int((y == 1).sum())
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Dataset Class Distribution"
        )

        counts = y.value_counts().sort_index()

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.bar(
            ["Healthy", "Parkinson's"],
            [
                counts.get(0, 0),
                counts.get(1, 0)
            ]
        )

        ax.set_ylabel(
            "Number of Records"
        )

        ax.set_title(
            "Class Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

    with right:

        st.subheader(
            "Model Overview"
        )

        st.metric(
            "Individual Models",
            5
        )

        st.metric(
            "Ensemble Models",
            3
        )

        st.metric(
            "Total Models",
            len(st.session_state.models)
        )

        st.metric(
            "Unique Subjects",
            df["subject_id"].nunique()
        )

    st.markdown("---")

    st.subheader(
        "Random Split Performance Overview"
    )

    st.dataframe(

        st.session_state.metrics[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC",
                "MCC"
            ]
        ],

        use_container_width=True,

        hide_index=True

    )

    st.caption(
        "Evaluation uses a stratified 80:20 random split."
    )


# =========================================================
# 22. PREDICTION
# =========================================================

elif page == "Prediction":

    st.subheader(
        "Voice Feature Input"
    )

    input_method = st.radio(

        "Choose Input Method",

        [
            "Manual Feature Entry",
            "CSV File Upload"
        ],

        horizontal=True

    )

    # ---------------------------------------------
    # MANUAL INPUT
    # ---------------------------------------------

    if input_method == "Manual Feature Entry":

        st.write(
            "Enter numerical voice feature values."
        )

        medians = X.median().fillna(0)

        with st.form(
            "manual_input_form"
        ):

            values = {}

            columns = st.columns(2)

            for i, feature in enumerate(
                feature_columns
            ):

                with columns[i % 2]:

                    values[feature] = st.number_input(

                        feature,

                        value=float(
                            medians[feature]
                        ),

                        format="%.6f",

                        key=f"manual_{feature}"

                    )

            submitted = st.form_submit_button(

                "Analyze Voice Features",

                use_container_width=True

            )

        if submitted:

            input_df = pd.DataFrame(

                [values],

                columns=feature_columns

            )

            st.session_state.latest_input = input_df

            st.session_state.latest_results = None

            st.session_state.latest_shap = None

            st.session_state.latest_source = "Manual Entry"

            st.success(
                "Manual input saved successfully."
            )

    # ---------------------------------------------
    # CSV INPUT
    # ---------------------------------------------

    else:

        st.write(
            "Upload a CSV containing the required voice features."
        )

        prediction_file = st.file_uploader(

            "Upload Prediction CSV",

            type=["csv"],

            key="prediction_csv"

        )

        if prediction_file is not None:

            try:

                uploaded_df = pd.read_csv(
                    prediction_file
                )

                uploaded_df.columns = (
                    uploaded_df.columns.astype(str).str.strip()
                )

                missing = [

                    feature

                    for feature in feature_columns

                    if feature not in uploaded_df.columns

                ]

                if missing:

                    st.error(
                        "Missing required features: "
                        + ", ".join(missing)
                    )

                elif uploaded_df.empty:

                    st.error(
                        "Uploaded CSV is empty."
                    )

                else:

                    input_df = uploaded_df[
                        feature_columns
                    ].copy()

                    for column in feature_columns:

                        input_df[column] = pd.to_numeric(

                            input_df[column],

                            errors="coerce"

                        )

                    input_df = input_df.replace(

                        [np.inf, -np.inf],

                        np.nan

                    )

                    if input_df.isna().all(axis=1).any():

                        st.error(
                            "At least one CSV record has no "
                            "usable numerical feature values."
                        )

                    else:

                        st.success(
                            f"{len(input_df)} records loaded."
                        )

                        st.dataframe(

                            input_df.head(10),

                            use_container_width=True

                        )

                        if st.button(

                            "Analyze CSV Records",

                            use_container_width=True,

                            key="analyze_csv"

                        ):

                            st.session_state.latest_input = input_df

                            st.session_state.latest_results = None

                            st.session_state.latest_shap = None

                            st.session_state.latest_source = "CSV Upload"

                            st.rerun()

            except Exception as error:

                st.error(
                    f"CSV error: {error}"
                )

    # ---------------------------------------------
    # SHARED PREDICTION PIPELINE
    # ---------------------------------------------

    saved_input = st.session_state.latest_input

    if saved_input is not None:

        st.markdown("---")

        st.subheader(
            "Prediction Results"
        )

        st.caption(
            f"Input Source: {st.session_state.latest_source}"
        )

        if st.session_state.latest_results is None:

            if st.button(

                "Run All Eight Models",

                use_container_width=True,

                key="run_prediction"

            ):

                try:

                    if saved_input.isna().all(axis=1).any():

                        raise ValueError(
                            "One or more records contain no usable features."
                        )

                    result_rows = []

                    summary_rows = []

                    progress = st.progress(0)

                    for i in range(len(saved_input)):

                        one_row = saved_input.iloc[
                            [i]
                        ].copy()

                        (

                            result_df,

                            predictions,

                            majority,

                            agreement,

                            pd_count,

                            healthy_count

                        ) = compare_models(

                            one_row,

                            st.session_state.models

                        )

                        result_df.insert(
                            0,
                            "Record",
                            i + 1
                        )

                        result_rows.append(
                            result_df
                        )

                        summary_rows.append({

                            "Record": i + 1,

                            "Majority Prediction": CLASS_NAMES[
                                majority
                            ],

                            "Parkinson's Votes": pd_count,

                            "Healthy Votes": healthy_count,

                            "Model Agreement (%)": agreement

                        })

                        progress.progress(
                            (i + 1) / len(saved_input)
                        )

                    progress.empty()

                    st.session_state.latest_results = {

                        "details": pd.concat(

                            result_rows,

                            ignore_index=True

                        ),

                        "summary": pd.DataFrame(
                            summary_rows
                        )

                    }

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Prediction error: {error}"
                    )

        # -----------------------------------------
        # DISPLAY RESULTS
        # -----------------------------------------

        if st.session_state.latest_results is not None:

            details = st.session_state.latest_results[
                "details"
            ]

            summary = st.session_state.latest_results[
                "summary"
            ]

            st.subheader(
                "Majority Model Results"
            )

            st.dataframe(

                summary,

                use_container_width=True,

                hide_index=True

            )

            st.subheader(
                "All Models Prediction and Confidence"
            )

            st.dataframe(

                details,

                use_container_width=True,

                hide_index=True

            )

            st.caption(
                "Hard Voting confidence is an estimated average "
                "of base-model probabilities, not a native calibrated "
                "probability of the Hard Voting classifier."
            )

            csv_result = details.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                "Download Prediction Results",

                data=csv_result,

                file_name="model_comparison_results.csv",

                mime="text/csv"

            )

            if len(summary) == 1:

                row = summary.iloc[0]

                st.markdown("---")

                st.subheader(
                    "Model Agreement"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Agreement",
                    f"{row['Model Agreement (%)']:.1f}%"
                )

                c2.metric(
                    "Parkinson's Votes",
                    int(row["Parkinson's Votes"])
                )

                c3.metric(
                    "Healthy Votes",
                    int(row["Healthy Votes"])
                )

                st.progress(
                    float(
                        row["Model Agreement (%)"]
                    ) / 100
                )

                st.info(
                    "Majority prediction: "
                    + row["Majority Prediction"]
                )

                st.subheader(
                    "Prediction Confidence Comparison"
                )

                chart_data = details.set_index(
                    "Model"
                )[
                    [
                        "Healthy Confidence (%)",
                        "Parkinson's Confidence (%)"
                    ]
                ]

                st.bar_chart(
                    chart_data,
                    use_container_width=True
                )

            else:

                st.info(
                    "Multiple records were evaluated. "
                    "Select a record in All Models Comparison "
                    "to inspect its individual results."
                )


# =========================================================
# 23. ALL MODELS COMPARISON
# =========================================================

elif page == "All Models Comparison":

    st.subheader(
        "All Models Comparison"
    )

    if st.session_state.latest_results is None:

        st.info(
            "Run a prediction first from the Prediction page."
        )

    else:

        details = st.session_state.latest_results[
            "details"
        ]

        summary = st.session_state.latest_results[
            "summary"
        ]

        selected_record = st.selectbox(

            "Select Input Record",

            summary["Record"].tolist()

        )

        record_results = details[

            details["Record"] == selected_record

        ].copy()

        summary_row = summary[

            summary["Record"] == selected_record

        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(

            "Model Agreement",

            f"{summary_row['Model Agreement (%)']:.1f}%"

        )

        c2.metric(

            "Parkinson's Votes",

            int(summary_row["Parkinson's Votes"])

        )

        c3.metric(

            "Healthy Votes",

            int(summary_row["Healthy Votes"])

        )

        st.markdown("---")

        st.subheader(
            "Prediction and Confidence Table"
        )

        st.dataframe(

            record_results.drop(
                columns=["Record"]
            ),

            use_container_width=True,

            hide_index=True

        )

        st.subheader(
            "Confidence Score Visualization"
        )

        chart = record_results.set_index(
            "Model"
        )[

            [
                "Healthy Confidence (%)",
                "Parkinson's Confidence (%)"
            ]

        ]

        st.bar_chart(
            chart,
            use_container_width=True
        )

        st.caption(
            "Confidence scores are model outputs and should not "
            "be interpreted as clinical certainty."
        )


# =========================================================
# 24. SHAP EXPLAINABILITY
# =========================================================

elif page == "SHAP Explainability":

    st.subheader(
        "SHAP Explainability"
    )

    st.write(
        "SHAP explains how individual voice features influence "
        "the Random Forest model output."
    )

    if st.session_state.latest_input is None:

        st.info(
            "Run a prediction first from the Prediction page."
        )

    else:

        latest_input = st.session_state.latest_input

        record_number = st.selectbox(

            "Select Record for Explanation",

            list(
                range(
                    1,
                    len(latest_input) + 1
                )
            )

        )

        selected_input = latest_input.iloc[
            [record_number - 1]
        ].copy()

        if st.button(

            "Generate SHAP Explanation",

            use_container_width=True

        ):

            try:

                with st.spinner(
                    "Generating SHAP explanation..."
                ):

                    shap_df = generate_shap_explanation(

                        selected_input,

                        feature_columns,

                        st.session_state.models[
                            "Random Forest"
                        ]

                    )

                    st.session_state.latest_shap = {

                        "record": record_number,

                        "data": shap_df

                    }

            except ImportError:

                st.error(
                    "SHAP is not installed. Run: pip install shap"
                )

            except Exception as error:

                st.error(
                    f"SHAP error: {error}"
                )

        saved_shap = st.session_state.latest_shap

        if (

            saved_shap is not None

            and saved_shap["record"] == record_number

        ):

            shap_df = saved_shap["data"]

            st.success(
                "SHAP explanation generated successfully."
            )

            st.markdown("---")

            st.subheader(
                "Important Voice Features"
            )

            top_features = shap_df.head(15).copy()

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            ax.barh(

                top_features["Feature"].iloc[::-1],

                top_features["SHAP Value"].iloc[::-1]

            )

            ax.axvline(
                0,
                linestyle="--",
                linewidth=1
            )

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_title(
                "Random Forest - SHAP Feature Impact"
            )

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "Positive SHAP values push the model output "
                "toward Parkinson's (class 1). Negative values "
                "push it toward Healthy (class 0)."
            )

            st.markdown("---")

            st.subheader(
                "SHAP Feature Impact Table"
            )

            st.dataframe(

                shap_df[

                    [
                        "Feature",
                        "Input Value",
                        "SHAP Value",
                        "Absolute Impact"
                    ]

                ],

                use_container_width=True,

                hide_index=True

            )

            st.markdown("---")

            st.subheader(
                "Top Influential Features"
            )

            for _, row in top_features.head(5).iterrows():

                if row["SHAP Value"] > 0:

                    direction = "toward Parkinson's"

                elif row["SHAP Value"] < 0:

                    direction = "toward Healthy"

                else:

                    direction = "neutral"

                st.write(

                    f"**{row['Feature']}** — "

                    f"{direction} "

                    f"(SHAP: {row['SHAP Value']:.5f})"

                )

            st.info(
                "SHAP explains the Random Forest model output. "
                "It does not provide a medical diagnosis."
            )


# =========================================================
# 25. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.subheader(
        "Random Split Model Evaluation"
    )

    st.caption(
        "Stratified 80:20 random train-test split."
    )

    selected_model_name = st.selectbox(

        "Select Model",

        MODEL_NAMES

    )

    model = st.session_state.models[
        selected_model_name
    ]

    X_test = st.session_state.X_test

    y_test = st.session_state.y_test

    predictions = model.predict(
        X_test
    )

    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

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

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(

            y_test,

            predictions,

            labels=[0, 1]

        )

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        ax.imshow(cm)

        ax.set_xticks([0, 1])

        ax.set_yticks([0, 1])

        ax.set_xticklabels([
            "Healthy",
            "Parkinson's"
        ])

        ax.set_yticklabels([
            "Healthy",
            "Parkinson's"
        ])

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        ax.set_title(
            "Confusion Matrix"
        )

        for i in range(2):

            for j in range(2):

                ax.text(

                    j,

                    i,

                    str(cm[i, j]),

                    ha="center",

                    va="center",

                    color="white"

                )

        st.pyplot(fig)

        plt.close(fig)

    with right:

        st.subheader(
            "Classification Report"
        )

        report = classification_report(

            y_test,

            predictions,

            labels=[0, 1],

            target_names=[
                "Healthy",
                "Parkinson's"
            ],

            output_dict=True,

            zero_division=0

        )

        st.dataframe(

            pd.DataFrame(
                report
            ).transpose(),

            use_container_width=True

        )

    # ROC curve

    if (

        hasattr(model, "predict_proba")

        and y_test.nunique() == 2

    ):

        try:

            probability = model.predict_proba(
                X_test
            )

            classes = list(
                model.classes_
            )

            if 1 in classes:

                positive_probability = probability[

                    :,

                    classes.index(1)

                ]

                fpr, tpr, _ = roc_curve(

                    y_test,

                    positive_probability

                )

                st.markdown("---")

                st.subheader(
                    "ROC Curve"
                )

                fig, ax = plt.subplots(
                    figsize=(6, 4)
                )

                ax.plot(

                    fpr,

                    tpr,

                    label=(
                        f"AUC = {metrics['ROC-AUC']:.4f}"
                    )

                )

                ax.plot(

                    [0, 1],

                    [0, 1],

                    linestyle="--"

                )

                ax.set_xlabel(
                    "False Positive Rate"
                )

                ax.set_ylabel(
                    "True Positive Rate"
                )

                ax.set_title(
                    "Receiver Operating Characteristic"
                )

                ax.legend()

                st.pyplot(fig)

                plt.close(fig)

        except Exception as error:

            st.warning(
                f"ROC curve unavailable: {error}"
            )


# =========================================================
# 26. SUBJECT-DISJOINT EVALUATION
# =========================================================

elif page == "Subject-Disjoint Evaluation":

    st.subheader(
        "Subject-Disjoint Evaluation"
    )

    st.write(
        "Five-fold GroupKFold evaluation. "
        "All five folds are evaluated independently."
    )

    st.caption(
        "Each fold uses different training and testing subjects. "
        "No subject is allowed to appear in both sets."
    )

    if st.button(

        "Run Five-Fold Subject-Disjoint Evaluation",

        use_container_width=True

    ):

        try:

            groups = (
                df["subject_id"]
                .astype(str)
                .reset_index(drop=True)
            )

            with st.spinner(
                "Running complete five-fold evaluation..."
            ):

                fold_df, summary_df = (
                    run_subject_disjoint_evaluation(
                        X.reset_index(drop=True),
                        y.reset_index(drop=True),
                        groups
                    )
                )

            st.session_state.subject_folds = fold_df

            st.session_state.subject_metrics = summary_df

            st.session_state.subject_info = {

                "records": len(df),

                "subjects": groups.nunique(),

                "folds": 5,

                "overlap": int(
                    fold_df["Subject Overlap"].sum()
                )

            }

            st.success(
                "All five folds completed successfully."
            )

        except Exception as error:

            st.error(
                f"Subject-disjoint evaluation failed: {error}"
            )

    # ---------------------------------------------
    # DISPLAY SUBJECT RESULTS
    # ---------------------------------------------

    if st.session_state.subject_metrics is not None:

        summary_df = st.session_state.subject_metrics

        fold_df = st.session_state.subject_folds

        info = st.session_state.subject_info

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Records",
            info["records"]
        )

        c2.metric(
            "Unique Subjects",
            info["subjects"]
        )

        c3.metric(
            "Evaluation Folds",
            info["folds"]
        )

        st.metric(
            "Total Subject Overlap",
            info["overlap"]
        )

        st.success(
            "Subject leakage check passed."
            if info["overlap"] == 0
            else "Subject overlap detected."
        )

        st.markdown("---")

        st.subheader(
            "Five-Fold Mean and Standard Deviation"
        )

        st.dataframe(

            summary_df,

            use_container_width=True,

            hide_index=True

        )

        st.markdown("---")

        st.subheader(
            "Fold-wise Evaluation Results"
        )

        selected_model = st.selectbox(

            "Select Model for Fold Comparison",

            MODEL_NAMES,

            key="subject_model_select"

        )

        selected_folds = fold_df[

            fold_df["Model"] == selected_model

        ].sort_values("Fold")

        st.dataframe(

            selected_folds,

            use_container_width=True,

            hide_index=True

        )

        st.markdown("---")

        st.subheader(
            "Subject-Disjoint Accuracy Comparison"
        )

        accuracy_chart = summary_df.set_index(
            "Model"
        )[
            [
                "Accuracy Mean",
                "Accuracy Std"
            ]
        ]

        st.bar_chart(
            accuracy_chart,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader(
            "Download Subject-Disjoint Results"
        )

        st.download_button(

            "Download Five-Fold Summary",

            data=summary_df.to_csv(
                index=False
            ).encode("utf-8"),

            file_name="subject_disjoint_summary.csv",

            mime="text/csv"

        )

        st.download_button(

            "Download Fold-wise Results",

            data=fold_df.to_csv(
                index=False
            ).encode("utf-8"),

            file_name="subject_disjoint_fold_results.csv",

            mime="text/csv"

        )


# =========================================================
# 27. MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.subheader(
        "Machine Learning Models"
    )

    individual = {

        "Logistic Regression":

            "A linear classification algorithm that estimates "
            "the relationship between voice features and the target class.",

        "SVM":

            "Support Vector Machine identifies a decision boundary "
            "between Healthy and Parkinson's classes.",

        "KNN":

            "K-Nearest Neighbors predicts a class based on "
            "nearby training samples.",

        "Decision Tree":

            "A tree-based classifier that makes predictions "
            "using feature-based decision rules.",

        "Random Forest":

            "An ensemble of decision trees that combines "
            "multiple tree predictions."

    }

    ensemble = {

        "Hard Voting":

            "Combines predicted classes from five base models "
            "using majority voting.",

        "Soft Voting":

            "Combines class probabilities from five base models "
            "and selects the class with the highest average probability.",

        "Stacking":

            "Uses predictions from five base models as input "
            "to a Logistic Regression meta-model."

    }

    st.markdown(
        "### Individual Models"
    )

    for name, description in individual.items():

        with st.expander(name):

            st.write(description)

    st.markdown(
        "### Ensemble Models"
    )

    for name, description in ensemble.items():

        with st.expander(name):

            st.write(description)

    st.markdown("---")

    st.subheader(
        "Dataset Details"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        len(df)
    )

    c2.metric(
        "Voice Features",
        len(feature_columns)
    )

    c3.metric(
        "Unique Subjects",
        df["subject_id"].nunique()
    )

    st.write(
        "Target Column: status"
    )

    st.write(
        "Class 0: Healthy"
    )

    st.write(
        "Class 1: Parkinson's Disease"
    )

    st.markdown("---")

    st.subheader(
        "System Capabilities"
    )

    capabilities = [

        "Manual voice feature entry",

        "CSV voice feature upload",

        "Eight-model prediction comparison",

        "Healthy and Parkinson's confidence scores",

        "Model agreement analysis",

        "SHAP-based feature explanation",

        "Random split evaluation",

        "Five-fold subject-disjoint evaluation",

        "Mean and standard deviation reporting",

        "Fold-wise performance comparison",

        "Downloadable evaluation results"

    ]

    for capability in capabilities:

        st.write(
            "✓ " + capability
        )


# =========================================================
# 28. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Parkinson's Disease Voice Classification System | "
    "Machine Learning and Explainable AI Research"
)

st.caption(
    "This research application is for experimental purposes "
    "and is not a substitute for professional medical diagnosis."
)
