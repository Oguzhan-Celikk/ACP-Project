import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from contextlib import asynccontextmanager

# Global variables for models
best_regressor = None
turbidity_classifier = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global best_regressor, turbidity_classifier, scaler
    try:
        print("Loading models...")
        best_regressor = joblib.load('best_regressor.pkl')
        turbidity_classifier = joblib.load('turbidity_classifier.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # In a real app, we might want to raise an error or exit, 
        # but for now we'll just log it. The endpoints will fail if models aren't loaded.
    yield

# Define the app
app = FastAPI(
    title="Water Level Prediction System",
    description="API for predicting water level and turbidity category based on sensor data.",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Input Schema
class SensorData(BaseModel):
    ir_value: float
    us_value: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyr_x: float
    gyr_y: float
    gyr_z: float

# Output Schema
class PredictionResponse(BaseModel):
    predicted_water_level: float
    detected_turbidity_status: str

@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    if not best_regressor or not turbidity_classifier or not scaler:
        raise HTTPException(status_code=500, detail="Models not loaded properly.")
    
    # Prepare input data
    # The scaler expects a 2D array with specific feature order
    features = [
        data.ir_value, data.us_value, 
        data.acc_x, data.acc_y, data.acc_z, 
        data.gyr_x, data.gyr_y, data.gyr_z
    ]
    
    input_array = np.array(features).reshape(1, -1)
    
    # Scale the input
    scaled_input = scaler.transform(input_array)
    
    # Predict Turbidity
    turbidity_pred = turbidity_classifier.predict(scaled_input)[0]
    
    # Predict Water Level
    # Note: The regressor was trained on the same scaled features but EXCLUDING the first one (IR value)
    # See pipeline.py: X_reg = X_scaled[:, 1:]
    water_level_input = scaled_input[:, 1:]
    water_level_pred = best_regressor.predict(water_level_input)[0]
    
    return {
        "predicted_water_level": float(water_level_pred),
        "detected_turbidity_status": str(turbidity_pred)
    }

@app.get("/")
async def root():
    return FileResponse('static/index.html')

def start_server():
    print('Starting Server...')       

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=True,
    )

if __name__ == "__main__":
    start_server()