import os
import io
import re

import matplotlib
matplotlib.use("Agg")

import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "mlops-youtube-comment-analysis"
    )
)


# --------------------------------------------------
# Flask
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# NLTK
# --------------------------------------------------

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english")) - {
    "not",
    "but",
    "however",
    "no",
    "yet"
}

lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_comment(comment):
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


# --------------------------------------------------
# Load model + vectorizer
# --------------------------------------------------

def load_model_and_vectorizer():

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI environment variable is not set."
        )

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = (
        "models:/youtube_comment_sentiment_model@candidate"
    )

    # Native LightGBM flavor
    model = mlflow.lightgbm.load_model(model_uri)

    vectorizer_path = os.path.join(
        PROJECT_DIR,
        "models",
        "tfidf_vectorizer.pkl"
    )

    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


model, vectorizer = load_model_and_vectorizer()


# --------------------------------------------------
# Health route
# --------------------------------------------------

@app.route("/")
def home():
    return "YouTube Comment Sentiment API is running"


# --------------------------------------------------
# Basic prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:

        cleaned_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        transformed_comments = vectorizer.transform(
            cleaned_comments
        )

        predictions = model.predict(
            transformed_comments
        )

        predictions = [
            str(pred)
            for pred in predictions
        ]

        response = [
            {
                "comment": comment,
                "sentiment": sentiment
            }
            for comment, sentiment
            in zip(comments, predictions)
        ]

        return jsonify(response)

    except Exception as e:

        return jsonify(
            {"error": f"Prediction failed: {str(e)}"}
        ), 500


# --------------------------------------------------
# Prediction with timestamps
# --------------------------------------------------

@app.route(
    "/predict_with_timestamps",
    methods=["POST"]
)
def predict_with_timestamps():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:

        comments = [
            item["text"]
            for item in comments_data
        ]

        timestamps = [
            item["timestamp"]
            for item in comments_data
        ]

        cleaned_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        transformed_comments = vectorizer.transform(
            cleaned_comments
        )

        predictions = model.predict(
            transformed_comments
        )

        predictions = [
            str(pred)
            for pred in predictions
        ]

        response = [
            {
                "comment": comment,
                "sentiment": sentiment,
                "timestamp": timestamp
            }
            for comment, sentiment, timestamp
            in zip(
                comments,
                predictions,
                timestamps
            )
        ]

        return jsonify(response)

    except Exception as e:

        return jsonify(
            {
                "error":
                f"Prediction failed: {str(e)}"
            }
        ), 500


# --------------------------------------------------
# Pie chart
# --------------------------------------------------

@app.route("/generate_chart", methods=["POST"])
def generate_chart():

    try:

        data = request.get_json()

        sentiment_counts = data.get(
            "sentiment_counts"
        )

        if not sentiment_counts:
            return jsonify(
                {"error": "No sentiment counts provided"}
            ), 400

        labels = [
            "Positive",
            "Neutral",
            "Negative"
        ]

        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0))
        ]

        if sum(sizes) == 0:
            return jsonify(
                {"error": "Sentiment counts are all zero"}
            ), 400

        plt.figure(figsize=(6, 6))

        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140
        )

        plt.axis("equal")
        plt.tight_layout()

        img_io = io.BytesIO()

        plt.savefig(
            img_io,
            format="PNG",
            transparent=True
        )

        img_io.seek(0)

        plt.close()

        return send_file(
            img_io,
            mimetype="image/png"
        )

    except Exception as e:

        app.logger.error(
            "Chart generation error: %s",
            e
        )

        return jsonify(
            {
                "error":
                f"Chart generation failed: {str(e)}"
            }
        ), 500


# --------------------------------------------------
# Word cloud
# --------------------------------------------------

@app.route(
    "/generate_wordcloud",
    methods=["POST"]
)
def generate_wordcloud():

    try:

        data = request.get_json()

        comments = data.get("comments")

        if not comments:
            return jsonify(
                {"error": "No comments provided"}
            ), 400

        cleaned_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        text = " ".join(cleaned_comments)

        if not text.strip():
            return jsonify(
                {"error": "No valid text available"}
            ), 400

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(
                stopwords.words("english")
            ),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()

        wordcloud.to_image().save(
            img_io,
            format="PNG"
        )

        img_io.seek(0)

        return send_file(
            img_io,
            mimetype="image/png"
        )

    except Exception as e:

        app.logger.error(
            "Word cloud error: %s",
            e
        )

        return jsonify(
            {
                "error":
                f"Word cloud generation failed: {str(e)}"
            }
        ), 500


# --------------------------------------------------
# Sentiment trend
# --------------------------------------------------

@app.route(
    "/generate_trend_graph",
    methods=["POST"]
)
def generate_trend_graph():

    try:

        data = request.get_json()

        sentiment_data = data.get(
            "sentiment_data"
        )

        if not sentiment_data:
            return jsonify(
                {"error": "No sentiment data provided"}
            ), 400

        df = pd.DataFrame(sentiment_data)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp"]
        )

        if df.empty:
            return jsonify(
                {"error": "No valid timestamps found"}
            ), 400

        df["sentiment"] = (
            df["sentiment"].astype(int)
        )

        df.set_index(
            "timestamp",
            inplace=True
        )

        monthly_counts = (
            df.resample("ME")["sentiment"]
            .value_counts()
            .unstack(fill_value=0)
        )

        monthly_totals = (
            monthly_counts.sum(axis=1)
        )

        monthly_percentages = (
            monthly_counts
            .div(monthly_totals, axis=0)
            * 100
        )

        for sentiment in [-1, 0, 1]:

            if sentiment not in monthly_percentages.columns:
                monthly_percentages[sentiment] = 0

        monthly_percentages = (
            monthly_percentages[
                [-1, 0, 1]
            ]
        )

        sentiment_labels = {
            -1: "Negative",
            0: "Neutral",
            1: "Positive"
        }

        plt.figure(figsize=(12, 6))

        for sentiment in [-1, 0, 1]:

            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment],
                marker="o",
                label=sentiment_labels[sentiment]
            )

        plt.title(
            "Monthly Sentiment Percentage Over Time"
        )

        plt.xlabel("Month")

        plt.ylabel(
            "Percentage of Comments (%)"
        )

        plt.grid(True)

        plt.xticks(rotation=45)

        plt.gca().xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )

        plt.gca().xaxis.set_major_locator(
            mdates.AutoDateLocator(
                maxticks=12
            )
        )

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()

        plt.savefig(
            img_io,
            format="PNG"
        )

        img_io.seek(0)

        plt.close()

        return send_file(
            img_io,
            mimetype="image/png"
        )

    except Exception as e:

        app.logger.error(
            "Trend graph error: %s",
            e
        )

        return jsonify(
            {
                "error":
                f"Trend graph generation failed: {str(e)}"
            }
        ), 500


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )