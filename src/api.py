import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.models.model import MyNeuralNet


classes = ["beagle", "bulldog", "dalmatian", "german-shepherd", 
           "husky", "labrador-retriever", "poodle", "rottweiler"]


app = FastAPI()
model = MyNeuralNet.load_from_checkpoint(lr=1e-4, checkpoint_path="models/model.ckpt")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/model")
async def cv_model(data: UploadFile = File(...)):
    if not data.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Invalid file type. Only images are allowed."}, status_code=400)
    
    content = await data.read()

    # TODO: Instead, it's better to pass the image to predict_model and let it do the 
    # resizing (and other preprocessing including normalization)
    img = cv2.imdecode(np.fromstring(content, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128)) # TODO: confirm this is actual model input shape
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    probs = torch.softmax(model(img), dim=1)
    prediction = classes[probs.argmax(dim=1).item()]
    confidence = probs.max(dim=1).values.item()

    return {"prediction": prediction, "confidence": confidence}