import os
import pickle

import mlflow
import pandas as pd
import pytest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


MODEL_NAME = (
    "youtube_comment_sentiment_model"
)

MODEL_ALIAS = "candidate"

TEST_DATA_PATH = (
    "data/interim/test_processed.csv"
)

VECTORIZER_PATH = (
    "models/tfidf_vectorizer.pkl"
)


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


def test_model_performance():

    try:

        model_uri = (
            f"models:/{MODEL_NAME}"
            f"@{MODEL_ALIAS}"
        )

        model = (
            mlflow.lightgbm.load_model(
                model_uri
            )
        )


        with open(
            VECTORIZER_PATH,
            "rb"
        ) as file:

            vectorizer = (
                pickle.load(file)
            )


        test_data = pd.read_csv(
            TEST_DATA_PATH
        )


        X_test = (
            test_data[
                "clean_comment"
            ]
            .fillna("")
        )


        y_test = (
            test_data["category"]
        )


        X_test_tfidf = (
            vectorizer.transform(
                X_test
            )
        )


        y_pred = model.predict(
            X_test_tfidf
        )


        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )


        print(
            f"Accuracy:  {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"F1:        {f1:.4f}"
        )


        expected_accuracy = 0.80
        expected_precision = 0.80
        expected_recall = 0.80
        expected_f1 = 0.80


        assert accuracy >= expected_accuracy

        assert precision >= expected_precision

        assert recall >= expected_recall

        assert f1 >= expected_f1


        print(
            "Model performance test passed."
        )


    except Exception as e:

        pytest.fail(
            f"Model performance test failed: {e}"
        )