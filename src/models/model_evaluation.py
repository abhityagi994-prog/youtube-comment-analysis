# src/models/model_evaluation.py

import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_evaluation_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_root_directory() -> str:
    """Return the project root directory."""

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    return os.path.abspath(
        os.path.join(current_dir, "../../")
    )


def load_data(file_path: str) -> pd.DataFrame:
    """Load test data."""

    try:
        df = pd.read_csv(file_path)

        df = df.dropna(
            subset=["clean_comment", "category"]
        )

        logger.debug(
            "Data loaded from %s",
            file_path
        )

        return df

    except Exception as e:
        logger.error(
            "Error loading data from %s: %s",
            file_path,
            e
        )
        raise


def load_model(model_path: str):
    """Load trained LightGBM model."""

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        logger.debug(
            "Model loaded from %s",
            model_path
        )

        return model

    except Exception as e:
        logger.error(
            "Error loading model from %s: %s",
            model_path,
            e
        )
        raise


def load_vectorizer(
    vectorizer_path: str
) -> TfidfVectorizer:

    """Load fitted TF-IDF vectorizer."""

    try:
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        logger.debug(
            "TF-IDF vectorizer loaded from %s",
            vectorizer_path
        )

        return vectorizer

    except Exception as e:
        logger.error(
            "Error loading vectorizer from %s: %s",
            vectorizer_path,
            e
        )
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from params.yaml."""

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug(
            "Parameters loaded from %s",
            params_path
        )

        return params

    except Exception as e:
        logger.error(
            "Error loading parameters from %s: %s",
            params_path,
            e
        )
        raise


def evaluate_model(model, X_test, y_test):
    """Generate predictions and evaluation results."""

    try:
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        logger.debug(
            "Model evaluation completed"
        )

        return accuracy, report, cm, y_pred

    except Exception as e:
        logger.error(
            "Error during model evaluation: %s",
            e
        )
        raise


def log_confusion_matrix(
    cm,
    output_path: str
) -> None:

    """Create and log confusion matrix."""

    try:
        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(
            "Confusion Matrix - Test Data"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()

        plt.savefig(
            output_path
        )

        mlflow.log_artifact(
            output_path,
            artifact_path="evaluation"
        )

        plt.close()

        logger.debug(
            "Confusion matrix logged"
        )

    except Exception as e:
        logger.error(
            "Error logging confusion matrix: %s",
            e
        )
        raise


def save_model_info(
    run_id: str,
    model_path: str,
    file_path: str
) -> None:

    """Save MLflow run information."""

    model_info = {
        "run_id": run_id,
        "model_path": model_path
    }

    with open(
        file_path,
        "w"
    ) as file:

        json.dump(
            model_info,
            file,
            indent=4
        )

    logger.debug(
        "Model info saved to %s",
        file_path
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    root_dir = get_root_directory()

    try:

        # ------------------------------------------
        # MLflow connection
        # ------------------------------------------

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI"
        )

        if not tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI environment variable is not set."
            )

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(tracking_uri)


        mlflow.set_experiment(
            "dvc-pipeline-runs"
        )


        with mlflow.start_run(
            run_name="LightGBM_DVC_Pipeline"
        ) as run:

            # --------------------------------------
            # Load parameters
            # --------------------------------------

            params = load_params(
                os.path.join(
                    root_dir,
                    "params.yaml"
                )
            )

            model_params = params[
                "model_building"
            ]

            mlflow.log_params(
                model_params
            )


            # --------------------------------------
            # Load model + vectorizer
            # --------------------------------------

            model = load_model(
                os.path.join(
                    root_dir,
                    "models",
                    "lgbm_model.pkl"
                )
            )

            vectorizer = load_vectorizer(
                os.path.join(
                    root_dir,
                    "models",
                    "tfidf_vectorizer.pkl"
                )
            )


            # --------------------------------------
            # Load test data
            # --------------------------------------

            test_data = load_data(
                os.path.join(
                    root_dir,
                    "data",
                    "interim",
                    "test_processed.csv"
                )
            )

            X_test_tfidf = vectorizer.transform(
                test_data["clean_comment"]
            )

            y_test = test_data[
                "category"
            ].values


            # --------------------------------------
            # Evaluate
            # --------------------------------------

            accuracy, report, cm, y_pred = evaluate_model(
                model,
                X_test_tfidf,
                y_test
            )

            logger.info(
                "Test accuracy: %.4f",
                accuracy
            )


            # --------------------------------------
            # MLflow metrics
            # --------------------------------------

            mlflow.log_metric(
                "test_accuracy",
                accuracy
            )

            for label, metrics in report.items():

                if isinstance(metrics, dict):

                    mlflow.log_metric(
                        f"test_{label}_precision",
                        metrics["precision"]
                    )

                    mlflow.log_metric(
                        f"test_{label}_recall",
                        metrics["recall"]
                    )

                    mlflow.log_metric(
                        f"test_{label}_f1_score",
                        metrics["f1-score"]
                    )


            # --------------------------------------
            # Model signature
            # --------------------------------------

            input_example = pd.DataFrame(
                X_test_tfidf[:5].toarray(),
                columns=vectorizer.get_feature_names_out()
            )

            predictions_example = model.predict(
                X_test_tfidf[:5]
            )

            signature = infer_signature(
                input_example,
                predictions_example
            )


            # --------------------------------------
            # Log LightGBM model
            # --------------------------------------

            mlflow.lightgbm.log_model(
                model,
                name="lgbm_model",
                signature=signature,
                input_example=input_example
            )


            # --------------------------------------
            # Log TF-IDF vectorizer
            # --------------------------------------

            mlflow.log_artifact(
                os.path.join(
                    root_dir,
                    "models",
                    "tfidf_vectorizer.pkl"
                ),
                artifact_path="vectorizer"
            )


            # --------------------------------------
            # Confusion matrix
            # --------------------------------------

            reports_dir = os.path.join(
                root_dir,
                "reports",
                "figures"
            )

            os.makedirs(
                reports_dir,
                exist_ok=True
            )

            cm_path = os.path.join(
                reports_dir,
                "confusion_matrix_final.png"
            )

            log_confusion_matrix(
                cm,
                cm_path
            )


            # --------------------------------------
            # Save experiment metadata
            # --------------------------------------

            experiment_info_path = os.path.join(
                root_dir,
                "experiment_info.json"
            )

            save_model_info(
                run.info.run_id,
                "lgbm_model",
                experiment_info_path
            )


            # --------------------------------------
            # Tags
            # --------------------------------------

            mlflow.set_tag(
                "model_type",
                "LightGBM"
            )

            mlflow.set_tag(
                "task",
                "Sentiment Analysis"
            )

            mlflow.set_tag(
                "dataset",
                "YouTube Comments"
            )

            mlflow.set_tag(
                "pipeline",
                "DVC"
            )


            print(
                f"Test Accuracy: {accuracy:.4f}"
            )

            print(
                classification_report(
                    y_test,
                    y_pred
                )
            )

            logger.debug(
                "Model evaluation stage completed successfully"
            )


    except Exception as e:

        logger.error(
            "Failed to complete model evaluation: %s",
            e
        )

        raise


if __name__ == "__main__":
    main()