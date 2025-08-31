# End-to-End MLOps

A  collection of **end-to-end MLOps projects**
This repository demonstrates how to build, train, track, and deploy a **models** using **DVC, MLflow, GitHub Actions, and AWS**.

---

## 📌 Workflows

1. Update `config.yaml`
2. Update `secrets.yaml` \[Optional]
3. Update `params.yaml`
4. Update the entity
5. Update the configuration manager in `src/config`
6. Update the components
7. Update the pipeline
8. Update the `main.py`
9. Update the `dvc.yaml`
10. Run `app.py`

---

## 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/Yusuf-Abol/endtoend-mlops.git
cd endtoend-mlops/projects/covid_chest_classifier
```

### STEP 1 — Create a Conda environment

```bash
conda create -n tubato_env python=3.11.13 -y
conda activate tubato_env
```

### STEP 2 — Install requirements

```bash
pip install -r requirements.txt
```

### STEP 3 — Run the app

```bash
python app.py
```

Now, open your local host and port in your browser.

---

## 📊 MLflow Integration

* [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
* [Tutorial Video](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

Run MLflow locally:

```bash
mlflow ui
```

Or connect with **DagsHub**:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<token>
```

---

## 🔄 DVC Commands

```bash
dvc init
dvc repro
dvc dag
```

---

## 📦 About MLflow & DVC

**MLflow**

* Production grade experiment tracking
* Logs, tags, and manages models

**DVC**

* Lightweight for experiments & pipelines
* Orchestrates ML workflows

---

## ☁️ AWS CICD Deployment with GitHub Actions

### 1. Login to AWS console

### 2. Create IAM user for deployment

* Access: **EC2 + ECR**
* Policies:

  * `AmazonEC2ContainerRegistryFullAccess`
  * `AmazonEC2FullAccess`

### 3. Create ECR repo

```bash
566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken
```

### 4. Create EC2 machine (Ubuntu)

Install Docker:

```bash
sudo apt-get update -y
sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 5. Configure EC2 as self-hosted GitHub Runner

Go to: `Settings > Actions > Runner > New Self-Hosted Runner`

### 6. Setup GitHub Secrets

```bash
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_ECR_LOGIN_URI=566373416292.dkr.ecr.ap-south-1.amazonaws.com
ECR_REPOSITORY_NAME=simple-app
```

---

✅ With this, you now have:

* **End-to-end pipeline** with configs + DVC
* **Experiment tracking** with MLflow
* **CI/CD pipeline** with GitHub Actions
* **Cloud deployment** with AWS


