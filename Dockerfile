FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app/ /app/flask_app/
COPY models/tfidf_vectorizer.pkl /app/models/tfidf_vectorizer.pkl

RUN python -m nltk.downloader stopwords wordnet

WORKDIR /app/flask_app

EXPOSE 5001

CMD ["python", "app.py"]