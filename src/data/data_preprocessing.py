# src/data/data_preprocessing.py

import pandas as pd
import os
import re
import nltk
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Download required NLTK resources
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


# Create once instead of once per comment
stop_words = set(stopwords.words("english")) - {
    "not",
    "but",
    "however",
    "no",
    "yet"
}

lemmatizer = WordNetLemmatizer()


def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""

    try:
        comment = comment.lower()

        comment = comment.strip()

        comment = re.sub(r"\n", " ", comment)

        comment = re.sub(
            r"[^A-Za-z0-9\s!?.,]",
            "",
            comment
        )

        comment = " ".join(
            word
            for word in comment.split()
            if word not in stop_words
        )

        comment = " ".join(
            lemmatizer.lemmatize(word)
            for word in comment.split()
        )

        return comment

    except Exception as e:
        logger.error(
            "Error while preprocessing comment: %s",
            e
        )
        raise


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text preprocessing to the dataframe."""

    try:
        df = df.copy()

        df["clean_comment"] = df["clean_comment"].apply(
            preprocess_comment
        )

        logger.debug("Text normalization completed")

        return df

    except Exception as e:
        logger.error(
            "Error during text normalization: %s",
            e
        )
        raise


def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_path: str
) -> None:

    """Save processed train and test datasets."""

    try:
        interim_data_path = os.path.join(
            data_path,
            "interim"
        )

        os.makedirs(
            interim_data_path,
            exist_ok=True
        )

        train_data.to_csv(
            os.path.join(
                interim_data_path,
                "train_processed.csv"
            ),
            index=False
        )

        test_data.to_csv(
            os.path.join(
                interim_data_path,
                "test_processed.csv"
            ),
            index=False
        )

        logger.debug(
            "Processed data saved to %s",
            interim_data_path
        )

    except Exception as e:
        logger.error(
            "Error while saving processed data: %s",
            e
        )
        raise


def main():

    try:
        logger.debug(
            "Starting data preprocessing..."
        )

        train_data = pd.read_csv(
            "./data/raw/train.csv"
        )

        test_data = pd.read_csv(
            "./data/raw/test.csv"
        )

        logger.debug(
            "Raw train and test data loaded"
        )

        train_processed_data = normalize_text(
            train_data
        )

        test_processed_data = normalize_text(
            test_data
        )

        save_data(
            train_processed_data,
            test_processed_data,
            data_path="./data"
        )

        logger.debug(
            "Data preprocessing completed successfully"
        )

    except Exception as e:
        logger.error(
            "Failed to complete data preprocessing: %s",
            e
        )

        raise


if __name__ == "__main__":
    main()