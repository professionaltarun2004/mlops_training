# ⚠️ Common MLOps Failures & Troubleshooting Guide

During the transition from a local machine learning script to a containerized production environment, several critical architectural and networking failures can occur. This document outlines the common pitfalls we encountered and how to resolve them.

---

## 1. The "File Protocol Trap" (Local Absolute Paths in Docker)

### The Failure
When loading a model inside a Docker container via `mlflow.sklearn.load_model("models:/TextClassifier@production")`, you might encounter a `FileNotFoundError` pointing to a local Windows path (e.g., `D:/MLOps/mlops_training/mlruns/...`).

### Why it Happens
When you train a model without specifying an explicit tracking URI and an S3-compatible artifact store, MLflow defaults to saving the absolute local file path in its SQLite database. When the Linux Docker container queries MLflow for the artifact location, it receives a Windows `D:/` path, which doesn't exist inside the isolated Linux container filesystem.

### The Fix
Use an S3-compatible remote storage system (like **MinIO**). Configure your `docker-compose.yml` with MinIO and pass the S3 credentials to both your training script and FastAPI container. This ensures the MLflow registry saves an S3 URI (`s3://mlflow-artifacts/...`) instead of a local file path.

---

## 2. MLflow Security Middleware (DNS Rebinding Attack)

### The Failure
When the API container attempts to communicate with the MLflow server over Docker Compose, it receives a `403 Forbidden` error:
`Invalid Host header - possible DNS rebinding attack detected`

### Why it Happens
In recent MLflow versions (3.x+), a security middleware enforces strict Host header validation. If your FastAPI container addresses the MLflow server via Docker Compose's service name (e.g., `http://mlflow:5000`), the Host header is `mlflow:5000`. By default, MLflow only accepts `localhost` or `127.0.0.1`.

### The Fix
Update your MLflow server startup command in `docker-compose.yml` to explicitly allow the hostname using the `--allowed-hosts` flag:
```bash
mlflow server --host 0.0.0.0 --allowed-hosts localhost,localhost:5000,mlflow,mlflow:5000
```

---

## 3. Disconnected Tracking URIs (Empty Registry)

### The Failure
You run your training script (`python -m src.train`), but when you check the MLflow UI running in Docker, it shows **0 experiments** and **0 models**.

### Why it Happens
Your training script on your local machine is defaulting to `./mlruns`. Meanwhile, the Dockerized MLflow server is using its own isolated SQLite database. You are effectively writing to a local ghost registry while viewing the empty Docker registry.

### The Fix
Explicitly set the Tracking URI at the very top of your training and model selection scripts so they talk to the active server:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

---

## 4. Unresolvable `host.docker.internal`

### The Failure
Using `host.docker.internal` as a hack to allow a Docker container to speak to a locally running Windows process (like MLflow).

### Why it Happens
While `host.docker.internal` works in Docker Desktop environments for development, it breaks completely in native Linux deployment environments (like AWS EC2 or Kubernetes). It tightly couples the containerized app to the host machine's specific networking state.

### The Fix
Move all interacting services into a unified **Docker Compose** network. The containers can then communicate safely using internal DNS service names (e.g., `http://mlflow:5000`), ensuring the architecture is 100% portable to the cloud.
