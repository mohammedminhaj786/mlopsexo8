Dynamic Traffic Prediction MLOps System
Overview

This project implements an end-to-end MLOps system for dynamic traffic congestion prediction.

The system uses machine learning models to predict traffic congestion based on traffic and environmental information. The complete workflow is automated using Apache Airflow, containerized using Docker, and deployed through a FastAPI REST API.

Objectives
Predict traffic congestion levels.
Compare multiple machine learning models.
Automatically select the best-performing model.
Automate the ML pipeline using Airflow.
Deploy the trained model using FastAPI.
Detect data drift.
Automatically retrain the model when drift is detected.
Monitor model performance and pipeline execution.
Handle failures through Airflow task management.
Maintain the complete project using GitHub.
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Apache Airflow
FastAPI
Uvicorn
Docker
Docker Compose
GitHub
Dataset

The dataset contains traffic and environmental information.

Features
Feature	Description
hour	Hour of the day
day_of_week	Day of the week
vehicle_count	Number of vehicles
average_speed	Average vehicle speed
occupancy	Road occupancy percentage
temperature	Temperature
congestion	Target congestion level
Target
0 → Low
1 → Medium
2 → High
Machine Learning Models

Three classification models are trained and compared:

Logistic Regression
Random Forest
Gradient Boosting

The models are evaluated using the weighted F1 Score.

Model Results
Model	F1 Score
Logistic Regression	0.595
Random Forest	0.652
Gradient Boosting	0.710
Best Model

Gradient Boosting

It achieved the highest F1 score of approximately 0.710 and was selected automatically by the pipeline.

MLOps Architecture
Traffic Dataset
       ↓
   Load Data
       ↓
 Data Validation
       ↓
 Train 3 Models
       ↓
 Model Evaluation
       ↓
 Select Best Model
       ↓
 Save Model
       ↓
 Data Drift Check
       ↓
 ┌───────────────┐
 │ Drift Detected│
 └───────┬───────┘
         ↓
   Automatic Retraining
         ↓
    Updated Model
         ↓
      FastAPI
         ↓
 Traffic Prediction
Airflow Pipeline

The Airflow DAG is:

traffic_prediction_pipeline

The pipeline contains:

load_data
     ↓
train_models
     ↓
check_drift_and_retrain
1. Load Data

Loads the traffic dataset and checks the available rows and columns.

2. Train Models

Trains:

Logistic Regression
Random Forest
Gradient Boosting

The models are compared using F1 score and the best model is saved.

3. Data Drift and Retraining

The system monitors traffic-related statistics such as:

Vehicle count
Average speed
Road occupancy

If significant changes are detected, the system automatically retrains the models and saves the new best model.

FastAPI Deployment

The trained model is deployed using FastAPI.

Start API

Docker Compose automatically starts the API.

API address:

http://localhost:8000
API Health Check
http://localhost:8000/
Prediction API

Example:

http://localhost:8000/predict?hour=18&day_of_week=1&vehicle_count=260&average_speed=14&occupancy=97&temperature=28

Example response:

{
    "prediction": 2,
    "congestion": "High"
}
Docker

The project uses Docker Compose to run:

Airflow
FastAPI

Check running containers:

docker compose ps

Start the system:

docker compose up -d

Stop the system:

docker compose down
Airflow

Airflow UI:

http://localhost:8080

Login:

Username: admin
Password: admin

DAG:

traffic_prediction_pipeline
Project Structure
mlops-exp8-traffic/
│
├── api/
│   └── app.py
│
├── airflow/
│   └── traffic_prediction_dag.py
│
├── config/
│
├── data/
│   └── traffic.csv
│
├── model/
│   ├── best_model.pkl
│   └── evaluation_results.csv
│
├── monitoring/
│
├── screenshots/
│
├── src/
│   ├── train.py
│   └── evaluate.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
Monitoring
Model-Level Monitoring

The system monitors:

F1 score
Model selection
Data drift
Prediction performance
System-Level Monitoring

The Docker and Airflow services can be monitored using:

docker compose ps

Airflow provides task status and execution logs.

FastAPI provides an endpoint for checking whether the prediction service is running.

Failure Handling
Failure Scenario 1 — Invalid or Missing Data

If the traffic dataset is missing or cannot be read, the Airflow task fails and the pipeline stops instead of producing an incorrect model.

Failure Scenario 2 — Data Drift

If traffic data changes significantly, the system detects drift and automatically retrains the model.

Failure Scenario 3 — API Failure

If the FastAPI container stops, Docker Compose shows the service as stopped and the API becomes unavailable. The container can be restarted using:

docker compose up -d
GitHub

Source code and project files are maintained in GitHub:

GitHub Repository – mlopsexo8

Conclusion

The Dynamic Traffic Prediction MLOps System demonstrates a complete machine learning lifecycle, starting from traffic data ingestion and model training to model selection, deployment, monitoring, data drift detection, and automated retraining.

The use of Docker, Apache Airflow, FastAPI, Scikit-learn, and GitHub provides a reproducible and automated MLOps workflow suitable for dynamic traffic prediction applications.
