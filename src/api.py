from fastapi import FastAPI
from fastapi import UploadFile, File
from typing import Optional
from src.models.model import MyNeuralNet
import torch
import cv2


app = FastAPI()
model = MyNeuralNet()
model = MyNeuralNet.load_from_checkpoint(checkpoint_path="model.ckpt")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.post("/model/")
async def cv_model(data: UploadFile = File(...)):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        img = cv2.imread("image.jpg")
        res = cv2.resize(img, (64, 64)) # TODO: confirm this is actual model input shape
        prediction = model(res)
        image.close()

    return prediction