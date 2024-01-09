import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
import wandb
from datetime import datetime

from models.model import MyNeuralNet
from data_loader import get_data # type: ignore

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

wandb_logger = pl.loggers.WandbLogger(
    name = wandb.api.viewer()["username"] + " - " + datetime.now().strftime("%Y-%m-%d %H:%M"), # run name format : username - timestamp 
    log_model=False, # if log_model == 'all', checkpoints are logged during training.
                     # if True, checkpoints are logged at the end of training, except when save_top_k == -1 which also logs every checkpoint during training.
                     # if log_model == False (default), no checkpoint is logged.
    project="Classification of dogbreeds",
    config={
        "architecture": "PretrainedVGG16",
        "dataset": "8DogBreeds",
        "learning_rate": lr,
        "batch_size" : batch_size,
        "max_epochs": max_epochs,
        "validation_ratio" : validation_ratio,
        "testing_ratio" : testing_ratio,
        "transforms" : "target_shape:"+str(target_shape)
    })


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
    trainer = pl.Trainer(logger=wandb_logger, log_every_n_steps=1, devices=1, accelerator=accelerator, max_epochs=max_epochs)

    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Test
    trainer.test(model, dataloaders=test_loader)

    # Save model
    trainer.save_checkpoint("src/models/checkpoints/model.ckpt")

    # if we want to save the model parameters only
    #torch.save(model.state_dict(), "model.pt")
    
