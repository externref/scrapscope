import fastapi

from src.predict import predict_image_bytes

app = fastapi.FastAPI()


@app.post("/predict")
async def predict(image: fastapi.UploadFile):
    image_bytes = await image.read()
    prediction = predict_image_bytes(image_bytes)
    return {"prediction": prediction}
