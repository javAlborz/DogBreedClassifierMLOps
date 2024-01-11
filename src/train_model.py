import os
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import wandb

from src.commiter.commiter import branch_and_commit
from src.data_loader import get_data  # type: ignore
from src.models.model import MyNeuralNet

BASE_DIR = os.getcwd()


@hydra.main(config_path=os.path.join(BASE_DIR, "src/conf"), config_name="training_config", version_base=None)
def main(cfg):
    commit_message = branch_and_commit()
    print("Commit message: ", commit_message)

    wandb_logger = pl.loggers.WandbLogger(
        name = wandb.api.viewer()["username"] + " - " + datetime.now().strftime("%Y-%m-%d %H:%M"), # run name format : username - timestamp 
        log_model=False, # if log_model == 'all', checkpoints are logged during training.
                        # if True, checkpoints are logged at the end of training, except when save_top_k == -1 which also logs every checkpoint during training.
                        # if log_model == False (default), no checkpoint is logged.
        project="Classification of dogbreeds",
        config={
            "architecture": "PretrainedVGG16",
            "dataset": "8DogBreeds",
            "learning_rate": cfg.lr,
            "batch_size" : cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "validation_ratio" : cfg.validation_ratio,
            "testing_ratio" : cfg.testing_ratio,
            "transforms" : "target_shape:"+str(tuple(cfg.target_shape)),
            "commit_message": commit_message
        })
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(tuple(cfg.target_shape),antialias=True)
        ])
    
    # Setup device
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Get model
    model = MyNeuralNet(cfg.lr)
    
    # Get data
    train_loader, valid_loader, test_loader = get_data(cfg.batch_size, cfg.validation_ratio, cfg.testing_ratio, transform=transform)

    # Setup trainer
    trainer = pl.Trainer(logger=wandb_logger, log_every_n_steps=1, devices=1, accelerator=accelerator, max_epochs=cfg.max_epochs)

    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Test
    trainer.test(model, dataloaders=test_loader)

    # Save model
    trainer.save_checkpoint("models/model.ckpt")

    # if we want to save the model parameters only
    #torch.save(model.state_dict(), "model.pt")
    return model, trainer


if __name__ == "__main__":
    main()
    
