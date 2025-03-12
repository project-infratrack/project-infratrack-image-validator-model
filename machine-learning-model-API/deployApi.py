from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the trained model
try:
    model = load_model('./model/potholemodel.h5')  # adjust path if needed.
    class_names = ['normal', 'potholes']  # adjust if needed.
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

img_size = 180

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")  # Ensure RGB
        img = img.resize((img_size, img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Rescaling.
        
        print("Shape of input_image_array:", img_array.shape)
        print("First 10 pixel values of input_image_array:", img_array[:10, :10, :]) # Print a small portion of the array

        print("Image preprocessed successfully.")

        prediction = model.predict(img_array)
        print(f"Raw prediction: {prediction}")

        result = np.argmax(prediction[0])
        predicted_class = class_names[result]

        print(f"Prediction: {predicted_class}")

        # Explicit if/else logic for pothole detection
        if predicted_class == 'potholes':
            pothole_detected = True
        elif predicted_class == 'normal':
            pothole_detected = False
        else: #Added to handle unexpected cases.
            pothole_detected = False #Default to false.

        return JSONResponse({"pothole_detected": pothole_detected, "predicted_class": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
