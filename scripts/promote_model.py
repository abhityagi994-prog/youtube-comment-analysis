import os

import mlflow
from mlflow import MlflowClient


MODEL_NAME = (
    "youtube_comment_sentiment_model"
)

SOURCE_ALIAS = "candidate"
TARGET_ALIAS = "champion"


def promote_model():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:

        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set"
        )


    mlflow.set_tracking_uri(
        tracking_uri
    )


    client = MlflowClient()


    candidate = (
        client
        .get_model_version_by_alias(
            name=MODEL_NAME,
            alias=SOURCE_ALIAS
        )
    )


    candidate_version = (
        candidate.version
    )


    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=TARGET_ALIAS,
        version=candidate_version
    )


    print(
        f"Model version "
        f"{candidate_version} "
        f"promoted from "
        f"'{SOURCE_ALIAS}' "
        f"to '{TARGET_ALIAS}'."
    )


if __name__ == "__main__":
    promote_model()