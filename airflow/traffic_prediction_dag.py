from datetime import datetime

import pandas as pd
import joblib

from airflow import DAG
from airflow.operators.python import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score


DATA_PATH = "/opt/airflow/data/traffic.csv"
MODEL_PATH = "/opt/airflow/model/best_model.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)

    print("Traffic data loaded successfully")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))


def train_models():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("congestion", axis=1)
    y = df["congestion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        score = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        print(name, "F1 Score:", score)

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print("Best Model:", best_name)
    print("Best F1 Score:", best_score)

    joblib.dump(best_model, MODEL_PATH)

    print("Best model saved successfully")


def check_drift_and_retrain():
    df = pd.read_csv(DATA_PATH)

    vehicle_mean = df["vehicle_count"].mean()
    speed_mean = df["average_speed"].mean()
    occupancy_mean = df["occupancy"].mean()

    print("Current vehicle count mean:", vehicle_mean)
    print("Current speed mean:", speed_mean)
    print("Current occupancy mean:", occupancy_mean)

    drift_detected = (
        vehicle_mean > 230
        or speed_mean < 20
        or occupancy_mean > 80
    )

    if drift_detected:
        print("WARNING: DATA DRIFT DETECTED")
        print("Starting automatic retraining...")

        X = df.drop("congestion", axis=1)
        y = df["congestion"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            )
        }

        best_model = None
        best_score = 0
        best_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            score = f1_score(
                y_test,
                predictions,
                average="weighted"
            )

            print("Retraining", name, "F1 Score:", score)

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        joblib.dump(best_model, MODEL_PATH)

        print("Automatic retraining completed")
        print("New Best Model:", best_name)
        print("New Best F1 Score:", best_score)

    else:
        print("No significant data drift detected")
        print("Retraining not required")


with DAG(
    dag_id="traffic_prediction_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )

    train_models_task = PythonOperator(
        task_id="train_models",
        python_callable=train_models
    )

    drift_task = PythonOperator(
        task_id="check_drift_and_retrain",
        python_callable=check_drift_and_retrain
    )

    load_data_task >> train_models_task >> drift_task