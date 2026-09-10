import os
import json
import logging

import mlflow
from mlflow import MlflowClient


tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

mlflow.set_tracking_uri(tracking_uri)

logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load run ID and model path from experiment_info.json."""

    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)

        logger.debug(
            "Model info loaded from %s",
            file_path
        )

        return model_info

    except FileNotFoundError:
        logger.error(
            "Model info file not found: %s",
            file_path
        )
        raise

    except Exception as e:
        logger.error(
            "Error loading model info: %s",
            e
        )
        raise

def register_model(model_name: str, model_info: dict):
    """Register the MLflow model and assign a candidate alias."""

    try:
        run_id = model_info["run_id"]
        model_path = model_info["model_path"]

        model_uri = f"runs:/{run_id}/{model_path}"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client = MlflowClient()

        client.set_registered_model_alias(
            name=model_name,
            alias="candidate",
            version=model_version.version
        )

        logger.info(
            "Registered model %s version %s with alias 'candidate'",
            model_name,
            model_version.version
        )

        return model_version

    except Exception as e:
        logger.error(
            "Model registration failed: %s",
            e
        )
        raise

def main():
    try:
        root_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../"
            )
        )

        model_info_path = os.path.join(
            root_dir,
            "experiment_info.json"
        )

        model_info = load_model_info(
            model_info_path
        )

        model_name = "youtube_comment_sentiment_model"

        model_version = register_model(
            model_name,
            model_info
        )

        print(
            f"Registered {model_name} "
            f"version {model_version.version} "
            f"with alias 'candidate'"
        )

    except Exception as e:
        logger.error(
            "Failed to complete model registration: %s",
            e
        )
        raise


if __name__ == "__main__":
    main()