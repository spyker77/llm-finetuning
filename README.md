# LLM Fine-tuning Project

This is an experimental attempt to create a production-ready pipeline for fine-tuning language models using LoRA (Low-Rank Adaptation) with MLflow tracking and Prefect orchestration.

## Project Overview

This project implements a complete workflow for fine-tuning the Qwen2.5-1.5B-Instruct model using LoRA, tracking experiments with MLflow, and serving the fine-tuned model via a FastAPI endpoint. The pipeline is orchestrated using Prefect and can be deployed on Kubernetes.

## Technologies Used

- **Python 3.12**
- **Qwen2.5-1.5B-Instruct**: Base language model
- **PEFT/LoRA**: Parameter-efficient fine-tuning technique
- **MLflow**: Experiment tracking and model registry
- **Prefect**: Workflow orchestration
- **FastAPI**: Inference API
- **Docker**: Containerization
- **Kubernetes**: Orchestration platform

## Project Structure

```bash
.
├── data
│   └── dataset.jsonl        # Training dataset with synthetic data
├── src
│   ├── finetuning           # Fine-tuning scripts
│   ├── inference            # Inference API
│   └── workflows            # Prefect workflow definitions
└── models                   # Saved model artifacts
```

## Setup Instructions

### Prerequisites

- Python 3.12+
- uv
- Docker
- Kubernetes cluster (optional)
- MLflow server

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run MLflow

```bash
# Inside activated environment, start MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

### 3. Run Fine-tuning with Prefect

```bash
# Start Prefect server
prefect server start

# In another terminal, create a work pool
prefect work-pool create process-pool --type process

# Start a worker
prefect worker start -p process-pool

# or
prefect worker start --pool k8s-pool --type kubernetes

# Run the fine-tuning flow in yet another terminal
python src/workflows/fine_tune_flow.py
```

### 4. Deploy to Kubernetes (Optional)

```bash
# Apply MLflow Helm chart
helm install mlflow community-charts/mlflow --namespace llm-finetuning --create-namespace

# Setup Self-hosted Prefect Server
helm install prefect-server prefect/prefect-server --namespace llm-finetuning

# Setup Worker for Self-hosted Prefect Server
helm install prefect-worker prefect/prefect-worker --namespace llm-finetuning -f prefect-worker-values.yaml

# Apply job to deploy workflow
kubectl apply -f prefect-deploy-job.yaml
```

## Common Commands

### Prefect Commands

```bash
# Apply all deployments from prefect.yaml
prefect deploy -n complete-pipeline-deployment

# Run a specific deployment
prefect deployment run 'LLM Pipeline/complete-pipeline-deployment'
```

### MLflow Commands

```bash
# View experiments
mlflow ui

# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Docker Commands

```bash
# Build fine-tuning image
docker build -f Dockerfile.finetuning -t llm-finetuning:latest .

# Build inference image
docker build -f Dockerfile.inference -t llm-inference:latest .

### For local registry
docker run -d -p 5000:5000 --restart always --name registry registry:2

# Build and push fine-tuning image
docker build -f Dockerfile.finetuning -t localhost:5000/llm-finetuning:latest .
docker push localhost:5000/llm-finetuning:latest

# Build and push inference image
docker build -f Dockerfile.inference -t localhost:5000/llm-inference:latest .
docker push localhost:5000/llm-inference:latest
```

## Future Improvements

- Add support for more base models
- Implement distributed training
- Add model evaluation metrics
- Implement A/B testing for deployed models

## License

MIT
