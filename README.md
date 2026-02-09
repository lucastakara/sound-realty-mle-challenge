# Sound Realty — Housing Price Prediction API 

This repository delivers a **real-time housing price prediction service** for Sound Realty (Seattle area).  
It exposes a **REST API** that receives JSON inputs, enriches requests on the backend using zipcode-level data, and returns a predicted sale price plus traceability metadata.

It also includes:
- **Baseline vs improved model evaluation** (5-fold cross-validation)
- **CatBoost upgrade + tuning**
- **Blue/Green zero-downtime deployment demo** using Docker Compose + Nginx
- **Automated metrics reporting** in Pull Requests via CML

---

## Contents
- [What this solves](#what-this-solves)
- [API overview](#api-overview)
- [Backend enrichment (zipcode demographics)](#backend-enrichment-zipcode-demographics)
- [Modeling approach](#modeling-approach)
- [Challenge requirements coverage](#challenge-requirements-coverage)
- [Tech stack](#tech-stack)
- [Quickstart (Conda)](#quickstart-conda)
- [Run locally (Docker Compose Blue/Green)](#run-locally-docker-compose-bluegreen)
- [Run live predictions on unseen examples](#run-live-predictions-on-unseen-examples)
- [Blue/Green deployment workflow](#bluegreen-deployment-workflow)
  
---
<a id="what-this-solves"></a>
## What this solves
Sound Realty spends significant time estimating property values manually. A proof-of-concept model existed, and the goal is to **deploy it for broader internal use**, while providing guidance and improvements to accuracy.

---
<a id="api-overview"></a>
## API overview

### `POST /predict` (Full input schema)
Accepts the same input columns as `data/future_unseen_examples.csv` (property attributes).  
**Demographics are not provided by the client**.

### `POST /predict_minimal` (Minimal input schema — bonus)
Accepts only the **minimum required property features** plus `zipcode`.  
The backend looks up demographics using the zipcode and completes the feature vector automatically.

### `GET /health`
Health endpoint used by tests and deployment checks. Returns API status and model metadata.

### Example response (both prediction endpoints)
Returns:
- `prediction` (price estimate)
- traceability metadata: `request_id`, `latency_ms`, `model_version`, `served_by`
- artifact paths for reproducibility

---
<a id="backend-enrichment-zipcode-demographics"></a>
## Backend enrichment (zipcode demographics)
Per the prompt requirement, **the API does not accept demographic fields** from the caller.  
Instead, the service:
1. Receives home attributes (plus `zipcode`)
2. Loads `data/zipcode_demographics.csv`
3. Merges demographics **server-side** using `zipcode`
4. Drops `zipcode` (used only as the join key)
5. Selects the ordered features from `model/model_features.json`
6. Calls `model.predict(...)`

This allows both `/predict` and `/predict_minimal` to produce a prediction using the same trained model feature set.

---

## Modeling approach
- **Baseline model:** K-Nearest Neighbors regressor (numeric feature vector)
- **Improved model:** CatBoost Regressor
  - hyperparameter tuning
  - cross-validation evaluation (5-fold)
  - produces substantially better MAE/RMSE and higher R² compared to baseline

Evaluation and metrics reporting are surfaced via:
- **CML (Continuous Machine Learning)** Pull Request comments

---

## Challenge requirements coverage

1) **Deploy model as REST endpoint receiving JSON POST data**  
- Implemented by `POST /predict` and `POST /predict_minimal`

2) **Inputs must match `future_unseen_examples.csv` and exclude demographics**  
- `/predict` matches the CSV schema (property attributes only)  
- Demographics are merged on the backend using zipcode

3) **Design for scaling and safe model updates**  
- Local demo implements **Blue/Green** with **Nginx** switching upstreams  
- This demonstrates a **zero-downtime promotion + instant rollback** pattern

4) **Bonus: minimal endpoint with only required features**  
- Implemented by `POST /predict_minimal`

5) **Test script that submits examples from `future_unseen_examples.csv`**  
- Implemented as a pytest-style live integration test + `make live` convenience target (see below)

---

## Tech stack
- **Python 3.x**
- **FastAPI** (REST API)
- **Scikit-learn / CatBoost** (training + inference)
- **Pandas** (data prep + merges)
- **Conda** (environment management)
- **Docker + Docker Compose** (local deployment)
- **Nginx** (reverse proxy + traffic switch for blue/green)
- **Pytest** (unit + integration tests)
- **GitHub Actions** (CI)
- **CML (Iterative)** (ML metrics reports in Pull Requests)

---

## Prerequisites (from scratch)

Install these tools before running the project:

- **Git** — clone the repository
- **Docker Desktop** — required to run Docker Engine + Docker Compose (used by `make up`)
- **Make** — used to run `make up`, `make down`, `make live`, etc.
- **Conda** — creates the Python environment used for local scripts/tests
- **Python** — installed via the Conda environment (no separate system Python required)

Verify installations:

```bash
git --version
docker --version
docker compose version
make --version
conda --version
```

## Quickstart (Conda)

Create and activate the environment (example):
```bash
conda env create -f conda_environment.yml
conda activate housing
```

## Run locally (Docker Compose Blue/Green)

Bring up the full stack (Nginx + Blue + Green):

```bash
make up
```

Check health:

```bash
curl -sS http://localhost:8000/health
```

Bring it down:
```
make down
```
What is running
*	api_blue and api_green run in parallel
* nginx is the single entrypoint on localhost:8000
* Nginx upstream can be switched to route traffic to Blue or Green (no downtime)

## Run live predictions on unseen examples
Starts the stack (if needed) and sends samples from data/future_unseen_examples.csv to the API:

```bash
make live
```

## Blue/Green deployment workflow

Run the Blue/Green end-to-end integration test:

```bash
make bluegreen
```
