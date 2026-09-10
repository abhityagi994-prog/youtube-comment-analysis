import os
import pickle

import mlflow
import pytest


MODEL_NAME = "youtube_comment_sentiment_model"
MODEL_ALIAS = "candidate"

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if not tracking_uri:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI is not set"
    )

mlflow.set_tracking_uri(tracking_uri)


def test_model_with_vectorizer():

    try:
        model_uri = (
            f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )

        model = mlflow.lightgbm.load_model(
            model_uri
        )

        with open(
            VECTORIZER_PATH,
            "rb"
        ) as file:

            vectorizer = pickle.load(file)

        input_text = [
            "This video is really helpful"
        ]

        input_tfidf = vectorizer.transform(
            input_text
        )

        prediction = model.predict(
            input_tfidf
        )

        assert (
            input_tfidf.shape[1]
            ==
            len(
                vectorizer
                .get_feature_names_out()
            )
        )

        assert len(prediction) == 1

        assert prediction[0] in [-1, 0, 1]

        print(
            "Model + vectorizer compatibility "
            "test passed."
        )

    except Exception as e:

        pytest.fail(
            f"Model/vectorizer test failed: {e}"
        )