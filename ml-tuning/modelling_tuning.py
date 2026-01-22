import matplotlib

matplotlib.use("Agg")

import os
from typing import Any, Dict, Tuple
import tempfile
import numpy as np
import pandas as pd
from hyperopt import space_eval
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("ml-diabetes-experiment")


BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing", "output"
)
DATASET_TRAIN_PATH = os.path.join(BASE_DIR, "diabetes_train.csv")
DATASET_TEST_PATH = os.path.join(BASE_DIR, "diabetes_test.csv")


SEARCH_SPACE = {
    "n_estimators": scope.int(hp.quniform("n_estimators", 10, 200, 10)),
    "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"]),
    "max_depth": hp.choice(
        "max_depth", [None, scope.int(hp.quniform("max_depth_int", 3, 15, 1))]
    ),
    "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
    "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 5, 1)),
    "max_features": hp.choice("max_features", ["sqrt", "log2"]),
    "class_weight": hp.choice("class_weight", [None, "balanced", "balanced_subsample"]),
}


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads training and testing datasets from CSV files and splits them into
    feature matrices and target vectors.

    Returns:
        result (Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
            A tuple containing:
            - X_train (pd.DataFrame): Feature set for the training data.
            - y_train (pd.Series): Target labels for the training data.
            - X_test (pd.DataFrame): Feature set for the testing data.
            - y_test (pd.Series): Target labels for the testing data.
    """
    train_df = pd.read_csv(DATASET_TRAIN_PATH, sep=",", encoding="utf-8")
    test_df = pd.read_csv(DATASET_TEST_PATH, sep=",", encoding="utf-8")

    X_train = train_df.drop(columns=["Outcome"])
    y_train = train_df["Outcome"]
    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]

    return X_train, y_train, X_test, y_test


def create_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, output_dir: str
) -> None:
    """
    Generates a confusion matrix heatmap and saves it as an image file.

    Args:
        y_true (pd.Series): True labels from the dataset.
        y_pred (np.ndarray): Predicted labels from the model.
        output_dir (str): Directory path where the image will be saved.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")

    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(cm_path)


def create_classification_report(
    y_true: pd.Series, y_pred: np.ndarray, output_dir: str
) -> None:
    """
    Generates a classification report and saves it as a text file.

    Args:
        y_true (pd.Series): True labels from the dataset.
        y_pred (np.ndarray): Predicted labels from the model.
        output_dir (str): Directory path where the text file will be saved.
    """
    report = classification_report(y_true, y_pred)
    report_path = os.path.join(output_dir, "test_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)


def create_feature_importance(
    model: Any, feature_names: pd.Index, output_dir: str
) -> None:
    """
    Extracts feature importance from the model and saves it as a CSV file.

    Args:
        model (Any): The trained model object (must have `feature_importances_` attribute).
        feature_names (pd.Index): Index containing the names of the features.
        output_dir (str): Directory path where the CSV file will be saved.
    """
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values(by="importance", ascending=False)

        fi_path = os.path.join(output_dir, "training_feature_importance.csv")
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)


def train_and_evaluate_model(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
) -> Dict[str, Any]:
    """
    Objective function for Hyperopt using Stratified K-Fold Cross Validation.

    Args:
        params (Dict[str, Any]): Dictionary of hyperparameters for the model.
        X_train (pd.DataFrame): Feature set for training.
        y_train (pd.Series): Target labels for training.
        n_splits (int): Number of splits for Cross Validation.

    Returns:
        result (Dict[str, Any]):
            A dictionary required by Hyperopt containing:
            - loss (float): Negative mean F1 score (to be minimized).
            - status (str): Status of the run (STATUS_OK).
            - metrics (Dict[str, float]): Dictionary of CV metrics.
    """
    scoring = ["accuracy", "f1", "roc_auc", "precision", "recall"]
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    mean_f1 = scores["test_f1"].mean()
    mean_accuracy = scores["test_accuracy"].mean()
    mean_precision = scores["test_precision"].mean()
    mean_recall = scores["test_recall"].mean()
    mean_roc_auc = scores["test_roc_auc"].mean()

    mlflow.log_params(params)
    mlflow.log_metric("cv_mean_f1", mean_f1)
    mlflow.log_metric("cv_mean_accuracy", mean_accuracy)
    mlflow.log_metric("cv_mean_precision", mean_precision)
    mlflow.log_metric("cv_mean_recall", mean_recall)
    mlflow.log_metric("cv_mean_roc_auc", mean_roc_auc)

    mlflow.set_tag("model", "RandomForest")
    mlflow.set_tag("optimization", "Hyperopt-TPE-CV")

    return {
        "loss": -mean_f1,
        "status": STATUS_OK,
        "metrics": {
            "cv_f1": mean_f1,
            "cv_accuracy": mean_accuracy,
            "cv_precision": mean_precision,
            "cv_recall": mean_recall,
            "cv_roc_auc": mean_roc_auc,
        },
    }


def train_and_evaluate_final_model(
    best_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Trains the final model using the best hyperparameters and evaluates it on the test
    set. Logs metrics, confusion matrix, and feature importance to MLflow.

    Args:
        best_params (Dict[str, Any]): The optimal hyperparameters found during tuning.
        X_train (pd.DataFrame): Feature set for training.
        y_train (pd.Series): Target labels for training.
        X_test (pd.DataFrame): Feature set for testing.
        y_test (pd.Series): Target labels for testing.
    """
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    mlflow.set_tag("stage", "production_candidate")
    mlflow.set_tag("optimization", "Hyperopt-TPE-CV")
    mlflow.set_tag("model", "RandomForest")

    mlflow.log_metrics(
        {
            "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_test_prob),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        create_confusion_matrix(y_test, y_test_pred, tmpdir)
        create_classification_report(y_test, y_test_pred, tmpdir)
        create_feature_importance(model, X_train.columns, tmpdir)


def main() -> None:
    """
    Main execution function.
    1. Loads data.
    2. Runs Hyperopt optimization with Cross Validation.
    3. Trains the final model with the best parameters and logs results.
    """

    # 1. Load Data
    X_train, y_train, X_test, y_test = load_data()

    # 2. Define Objective Function (Hyperopt + CV)
    def objective(params):
        with mlflow.start_run(nested=True):
            return train_and_evaluate_model(params, X_train, y_train, n_splits=6)

    # 3. Run Hyperopt Tuning
    print("Starting Hyperopt Tuning...")
    with mlflow.start_run(run_name="Hyperopt Optimization CV") as run:
        mlflow.set_tag("stage", "tuning")

        trials = Trials()
        best = fmin(
            fn=objective,
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            rstate=np.random.default_rng(42),
        )
        best_params = space_eval(SEARCH_SPACE, best)
        print(f"Best Params: {best_params}")

    # 4. Train Best Model (Final) with Autolog
    print("Starting Final Model Training with Autolog...")

    # Enable Autolog only for this final run
    mlflow.sklearn.autolog(log_input_examples=True)

    with mlflow.start_run(run_name="Best Model Training"):
        train_and_evaluate_final_model(best_params, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
