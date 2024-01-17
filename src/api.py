import os
import tempfile

import torch
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.transforms import ToTensor

from src.models.model import MyNeuralNet


def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Apply the same transformations as in training
    transform = ToTensor()
    image = transform(image)

    return image

with open('src/conf/training_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

lr = config['lr']

checkpoint_path = "models/model.ckpt"

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

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(content)
        temp_path = temp.name

    # Preprocess the image
    img_tensor = preprocess_image(temp_path)

    # Add an extra dimension for batch size
    img_tensor = img_tensor.unsqueeze(0)

    # Make sure the image is on the same device as the model
    img_tensor = img_tensor.to(device)

    probs = torch.softmax(model(img_tensor), dim=1)
    prediction = classes[int(probs.argmax(dim=1).item())]
    confidence = probs.max(dim=1).values.item()

    # Delete the temporary file
    os.remove(temp_path)

    return {"prediction": prediction, "confidence": confidence}