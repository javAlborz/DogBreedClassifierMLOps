import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import yaml
from models.model import MyNeuralNet

with open('conf/training_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

lr = config['lr']

checkpoint_path = "../models/model.ckpt"

classes = ["beagle", "bulldog", "dalmatian", "german-shepherd", 
           "husky", "labrador-retriever", "poodle", "rottweiler"]


app = FastAPI()
model = MyNeuralNet.load_from_checkpoint(lr=lr, checkpoint_path=checkpoint_path)
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
    img = cv2.imdecode(np.fromstring(content, dtype=np.uint8), cv2.IMREAD_COLOR)    # type: ignore
    img = cv2.resize(img, (128, 128))   # TODO: confirm this is actual model input shape
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    probs = torch.softmax(model(img_tensor), dim=1)
    prediction = classes[int(probs.argmax(dim=1).item())]
    confidence = probs.max(dim=1).values.item()

    return {"prediction": prediction, "confidence": confidence}