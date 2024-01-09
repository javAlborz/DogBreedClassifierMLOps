import torch
import torchvision.transforms as transforms
import lightning as L

from models.model import MyNeuralNet
from data_loader import get_data

# Config 
lr = 1e-4 

batch_size = 32
max_epochs = 10

validation_ratio=0.15
testing_ratio=0.10

target_shape = (128,128) 
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_shape,antialias=True)
    ])


if __name__ == "__main__":
    # Setup device
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Get model
    model = MyNeuralNet(lr)

    # Get data
    train_loader, valid_loader, test_loader = get_data(batch_size, validation_ratio, testing_ratio, transform=transform)

    # Setup trainer
    trainer = L.Trainer(devices=1, accelerator=accelerator, max_epochs=max_epochs)

    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Test
    trainer.test(model, dataloaders=test_loader)

    # Save model
    trainer.save_checkpoint("src/models/checkpoints/model.ckpt")

    # if we want to save the model parameters only
    #torch.save(model.state_dict(), "model.pt")
    
