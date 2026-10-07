from fastapi import FastAPI
import joblib

app = FastAPI(title="Traffic Prediction API")

model = joblib.load("model/best_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Dynamic Traffic Prediction API is running"
    }


@app.get("/predict")
def predict(
    hour: int,
    day_of_week: int,
    vehicle_count: int,
    average_speed: float,
    occupancy: float,
    temperature: float
):

    data = [[
        hour,
        day_of_week,
        vehicle_count,
        average_speed,
        occupancy,
        temperature
    ]]

    prediction = model.predict(data)[0]

    if prediction == 0:
        congestion = "Low"
    elif prediction == 1:
        congestion = "Medium"
    else:
        congestion = "High"

    return {
        "prediction": int(prediction),
        "congestion": congestion
    }