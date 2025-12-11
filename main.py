
# # 1️⃣ Import Libraries
# ✔ FastAPI → used to create API endpoints (GET, POST)
# ✔ numpy → convert input features into arrays for ML model
# ✔ joblib → load the saved scaler (scaler.pkl)
# ✔ tensorflow → load the deep learning model
# ✔ pydantic → validate incoming JSON data

from fastapi import FastAPI
import numpy as np
import joblib
import tensorflow as tf
from pydantic import BaseModel

# 2️⃣ Create FastAPI App
# This creates your API application.
# When you run Uvicorn, this app will be served.

app = FastAPI()

# 3️⃣ Load Model and Scaler
# ✔ Loads the pretrained breast cancer deep learning model
# ✔ Loads the standard scaler used during training


model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")

#4️⃣ Define Input Format (Pydantic Model)
class InputData(BaseModel):
    features: list

# 5️⃣ GET Request – Test API
@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is working!"}

# 6️⃣ POST Request – Prediction Endpoint
@app.post("/predict")
def predict(data: InputData):
# ✔ This endpoint accepts POST requests
# ✔ It receives JSON that matches InputData
    
    # 7️⃣ Convert Input to Numpy Array
    input_array = np.asarray(data.features)

    # 8️⃣ Reshape for Single Sample
    input_reshaped = input_array.reshape(1, -1)

    # 9️⃣Standardize using training scaler
    input_std = scaler.transform(input_reshaped)

    # 🔟 Predict probability
    prediction = model.predict(input_std)

    # 1️⃣1️⃣Convert to class label
    predicted_label = int(np.argmax(prediction))
    
    # 1️⃣2️⃣ Return JSON Response
    # ✔ Convert the prediction array to a normal Python list
    # ✔ Return both probabilities and the predicted class
    return {
        "probabilities": prediction.tolist(),
        "class_label": predicted_label
    }
