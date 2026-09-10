import os

import mlflow
import pytest


MODEL_NAME = "youtube_comment_sentiment_model"
MODEL_ALIAS = "candidate"

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if not tracking_uri:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(tracking_uri)


def test_load_candidate_model():
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        model = mlflow.lightgbm.load_model(model_uri)

        assert model is not None

        print(
            f"Model '{MODEL_NAME}' "
            f"with alias '{MODEL_ALIAS}' "
            f"loaded successfully."
        )

    except Exception as e:
        pytest.fail(
            f"Model loading failed: {e}"
        )