# src/models/model_building.py

import os
import pickle
import yaml
import logging

import pandas as pd
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer


# Logging configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def get_root_directory() -> str:
    """Return the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(current_dir, "../../")
    )


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

    except FileNotFoundError:
        logger.error(
            "Parameter file not found: %s",
            params_path
        )
        raise

    except yaml.YAMLError as e:
        logger.error(
            "YAML parsing error: %s",
            e
        )
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load processed training data."""
    try:
        df = pd.read_csv(file_path)

        df = df.dropna(
            subset=["clean_comment", "category"]
        )

        logger.debug(
            "Training data loaded from %s",
            file_path
        )

        return df

    except Exception as e:
        logger.error(
            "Error loading training data: %s",
            e
        )
        raise


def apply_tfidf(
    train_data: pd.DataFrame,
    max_features: int,
    ngram_range: tuple
):
    """Fit TF-IDF on training comments."""

    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

        X_train = train_data["clean_comment"]
        y_train = train_data["category"]

        X_train_tfidf = vectorizer.fit_transform(
            X_train
        )

        logger.debug(
            "TF-IDF complete. Shape: %s",
            X_train_tfidf.shape
        )

        return X_train_tfidf, y_train, vectorizer

    except Exception as e:
        logger.error(
            "TF-IDF transformation failed: %s",
            e
        )
        raise


def train_lgbm(
    X_train,
    y_train,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    reg_alpha: float,
    reg_lambda: float
) -> lgb.LGBMClassifier:

    """Train the final LightGBM classifier."""

    try:
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",

            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,

            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,

            random_state=42,
            verbosity=-1
        )

        model.fit(
            X_train,
            y_train
        )

        logger.debug(
            "LightGBM model training completed"
        )

        return model

    except Exception as e:
        logger.error(
            "LightGBM training failed: %s",
            e
        )
        raise


def save_pickle(obj, file_path: str) -> None:
    """Save a Python object using pickle."""

    try:
        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

        logger.debug(
            "Saved object to %s",
            file_path
        )

    except Exception as e:
        logger.error(
            "Failed to save object: %s",
            e
        )
        raise


def main():

    try:
        root_dir = get_root_directory()

        params = load_params(
            os.path.join(
                root_dir,
                "params.yaml"
            )
        )

        model_params = params["model_building"]

        max_features = model_params["max_features"]

        ngram_range = tuple(
            model_params["ngram_range"]
        )

        learning_rate = model_params["learning_rate"]
        max_depth = model_params["max_depth"]
        n_estimators = model_params["n_estimators"]

        reg_alpha = model_params["reg_alpha"]
        reg_lambda = model_params["reg_lambda"]

        train_data = load_data(
            os.path.join(
                root_dir,
                "data/interim/train_processed.csv"
            )
        )

        X_train_tfidf, y_train, vectorizer = apply_tfidf(
            train_data,
            max_features,
            ngram_range
        )

        model = train_lgbm(
            X_train_tfidf,
            y_train,
            learning_rate,
            max_depth,
            n_estimators,
            reg_alpha,
            reg_lambda
        )

        models_dir = os.path.join(
            root_dir,
            "models"
        )

        save_pickle(
            vectorizer,
            os.path.join(
                models_dir,
                "tfidf_vectorizer.pkl"
            )
        )

        save_pickle(
            model,
            os.path.join(
                models_dir,
                "lgbm_model.pkl"
            )
        )

        logger.debug(
            "Model building stage completed successfully"
        )

    except Exception as e:
        logger.error(
            "Model building stage failed: %s",
            e
        )
        raise


if __name__ == "__main__":
    main()