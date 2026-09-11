# YouTube Comment Sentiment Analysis

An end-to-end YouTube comment sentiment analysis system with a Chrome
extension interface, Flask inference API, reproducible ML pipeline,
experiment tracking, automated model validation, containerization, and
AWS deployment.

The project extends beyond model training by implementing an MLOps
workflow using **DVC, MLflow, GitHub Actions, Docker, AWS S3, Amazon
ECR, and Amazon ECS/Fargate**.

## Overview

The system analyzes YouTube comments and classifies sentiment into three
categories:

- **Positive:** `1`
- **Neutral:** `0`
- **Negative:** `-1`

A Chrome extension provides the user-facing interface. A Flask API
handles preprocessing and inference using a **TF-IDF vectorizer** and
**LightGBM classifier**.

## Key Features

- Three-class YouTube comment sentiment classification
- Chrome extension interface
- Flask REST API
- NLTK text preprocessing
- TF-IDF feature extraction
- LightGBM classification
- Reproducible DVC pipeline
- MLflow experiment tracking and Model Registry
- `candidate` → `champion` model alias workflow
- Automated model loading, compatibility, and performance tests
- GitHub Actions continuous integration
- DVC/MLflow artifact storage on Amazon S3
- Dockerized inference application
- Docker image storage on Amazon ECR
- Deployment on Amazon ECS/Fargate
- ECS health checks and CloudWatch logging

## Application Architecture

``` text
YouTube Video
     |
     v
Chrome Extension
     |
     v
Flask REST API
     |
     v
Text Preprocessing
     |
     v
TF-IDF Vectorizer
     |
     v
LightGBM Model
     |
     v
Sentiment (-1 / 0 / 1)
     |
     v
Chrome Extension
```

## MLOps Architecture

``` text
Data
 |
 v
DVC Pipeline
 |
 v
Training / Evaluation
 |
 v
MLflow Experiment Tracking
 |
 v
Model Registry
 |
 v
candidate
 |
 v
GitHub Actions
 |
 +--> Model Loading Test
 +--> Model/Vectorizer Compatibility Test
 +--> Model Performance Test
 |
 v
champion
 |
 v
Docker
 |
 v
Amazon ECR
 |
 v
Amazon ECS / Fargate
 |
 v
Flask Inference API
```

## Model Lifecycle

The registry uses MLflow aliases rather than the legacy
Staging/Production stage workflow.

``` text
New Model
   |
   v
candidate
   |
   v
Automated CI Validation
   |
 +---+---+
 |       |
Fail    Pass
 |       |
Stop     v
      champion
```

## Tech Stack

| Area                           | Technologies             |
|--------------------------------|--------------------------|
| Language                       | Python                   |
| Data Processing                | Pandas, NumPy            |
| Machine Learning               | Scikit-learn, LightGBM   |
| NLP                            | NLTK, TF-IDF             |
| API                            | Flask, Flask-CORS        |
| Visualization                  | Matplotlib, WordCloud    |
| Experiment Tracking            | MLflow                   |
| Model Registry                 | MLflow Model Registry    |
| Pipeline / Artifact Versioning | DVC                      |
| Testing                        | Pytest                   |
| CI                             | GitHub Actions           |
| Cloud Storage                  | Amazon S3                |
| Containerization               | Docker                   |
| Container Registry             | Amazon ECR               |
| Deployment                     | Amazon ECS / AWS Fargate |
| Logging                        | Amazon CloudWatch        |
| Cloud Security                 | AWS IAM, Security Groups |
| Version Control                | Git, GitHub              |

## Project Structure

``` text
.
├── .dvc/
├── .github/
│   └── workflows/
│       └── ci.yaml
├── data/
├── flask_app/
│   └── app.py
├── models/
│   └── tfidf_vectorizer.pkl
├── scripts/
│   ├── promote_model.py
│   ├── test_load_model.py
│   ├── test_model_performance.py
│   └── test_model_signature.py
├── src/
│   └── models/
│       └── register_model.py
├── yt-chrome-plug-in/
│   ├── manifest.json
│   ├── popup.html
│   └── popup.js
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── requirements-freeze.txt
└── README.md
```

> Generated artifacts, caches, virtual environments, and
> environment-specific files are omitted from this simplified structure.

## Running Locally

``` bash
git clone <repository-url>
cd mlops-youtube-comment-analysis
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment on Windows:

``` powershell
.venv\Scripts\Activate.ps1
```

If valid AWS credentials for the configured DVC remote are available:

``` bash
dvc pull
```

Set the MLflow tracking URI:

``` powershell
$env:MLFLOW_TRACKING_URI="<your-mlflow-tracking-uri>"
```

Run the Flask API:

``` bash
python flask_app/app.py
```

The API runs locally on port `5001`.

## API

### Health Check

``` http
GET /
```

Expected response:

``` text
YouTube Comment Sentiment API is running
```

### Predict Sentiment

``` http
POST /predict
Content-Type: application/json
```

Example request:

``` json
{
  "comments": [
    "I absolutely love this video, amazing work!",
    "This is the worst video I have ever watched.",
    "The video was uploaded yesterday."
  ]
}
```

Example response:

``` json
[
  {"comment": "I absolutely love this video, amazing work!", "sentiment": "1"},
  {"comment": "This is the worst video I have ever watched.", "sentiment": "-1"},
  {"comment": "The video was uploaded yesterday.", "sentiment": "0"}
]
```

## DVC Pipeline

DVC defines and reproduces the machine learning workflow:

``` bash
dvc repro
```

DVC tracks stage dependencies and outputs, while DVC-managed artifacts
are stored remotely in Amazon S3.

## MLflow

MLflow is used for experiment tracking and model management. Validated
models are registered under:

``` text
youtube_comment_sentiment_model
```

The workflow uses `candidate` and `champion` aliases instead of
hard-coded model version numbers.

## Automated Testing

The CI workflow validates candidate models through:

- `test_load_model.py` — verifies that the candidate model can be
  loaded.
- `test_model_signature.py` — verifies model/vectorizer compatibility.
- `test_model_performance.py` — verifies that configured performance
  thresholds are met.
- `promote_model.py` — updates the `champion` alias after successful
  validation.

Run tests locally with:

``` bash
python -m pytest -v
```

## Continuous Integration

GitHub Actions automates dependency installation, DVC pipeline
reproduction, artifact synchronization, model validation, and model
promotion.

``` text
Git Push
   |
   v
GitHub Actions
   |
   v
Install Dependencies
   |
   v
DVC Reproduction
   |
   v
Model Registration
   |
   v
Automated Tests
   |
   v
Promote Valid Candidate
```

AWS credentials used during active development were supplied through
**GitHub Secrets**, not committed to the repository.

> **Infrastructure status:** Temporary AWS infrastructure and project
> credentials used for deployment validation were intentionally
> decommissioned after successful end-to-end testing to avoid
> unnecessary cloud costs. Historical successful CI runs remain in the
> repository history. Rerunning cloud-dependent workflow steps requires
> valid AWS credentials and infrastructure to be configured again.

## Docker

Build the image:

``` bash
docker build -t youtube-comment-analysis .
```

The container exposes the Flask application on port `5001`.

``` text
Host :5001  --->  Container :5001
```

Runtime AWS credentials should be provided through an appropriate
credential mechanism and must not be embedded in the image.

## AWS Deployment

The containerized API was deployed to AWS to validate the complete
deployment workflow.

``` text
Source Code
    |
    v
Docker Image
    |
    v
Amazon ECR
    |
    v
Amazon ECS / Fargate
    |
    v
Public Ingress
    |
    v
Flask Sentiment API
```

The ECS task pulled the private image from ECR and ran the Flask service
on port `5001`. IAM task roles were used for runtime AWS access rather
than embedding permanent credentials in the container.

ECS health checks and CloudWatch logs verified that the container was
running and returning successful HTTP responses. The public ingress
endpoint was successfully validated.

After end-to-end deployment validation, temporary ECS/Fargate, ECR, EC2,
load-balancer, deployment-specific security, and MLflow artifact
resources were decommissioned to avoid unnecessary cloud costs. The DVC
remote was retained separately in Amazon S3.

## Chrome Extension

The extension is contained in:

``` text
yt-chrome-plug-in/
├── manifest.json
├── popup.html
└── popup.js
```

It interacts with YouTube and the sentiment API to analyze comments for
the active video.

Sensitive API credentials should not be committed directly into
`popup.js`.

## Reproducibility

- Git/GitHub tracks source code and configuration.
- DVC tracks pipeline outputs and large artifacts.
- Amazon S3 acts as the DVC remote.
- `dvc.yaml` defines reproducible pipeline stages.
- `params.yaml` separates pipeline parameters from code.
- MLflow records experiments and registered model versions.
- `requirements.txt` documents direct dependencies.
- `requirements-freeze.txt` records the resolved development
  environment.
- Docker provides a consistent inference runtime.

## What This Project Demonstrates

``` text
NLP Preprocessing
       +
Model Training
       +
Experiment Tracking
       +
Reproducible Pipelines
       +
Model Registry
       +
Automated Validation
       +
Continuous Integration
       +
Containerization
       +
Cloud Artifact Storage
       +
Managed Container Deployment
```

The project demonstrates the complete path from NLP model development to
a containerized cloud inference service, including practical debugging
across dependency management, IAM permissions, remote MLflow
connectivity, CI runners, Docker runtime configuration, ECS networking,
and health checks.

## Future Improvements

- Secure HTTPS endpoint and production authentication
- API rate limiting
- Automated ECS continuous deployment
- Infrastructure as Code with Terraform or AWS CDK
- Application and model monitoring
- Data/prediction drift detection
- Automated retraining
- AWS Secrets Manager integration
- Expanded API and end-to-end tests
- Scalable asynchronous comment processing

## License

This project is available under the repository’s MIT License.
