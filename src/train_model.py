import os
import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from src.data_loader import get_data  
from src.models.model import MyNeuralNet

BASE_DIR = os.getcwd()

os.environ["MIN_TEST_LOSS"] = str(100000000.0)

@hydra.main(config_path=os.path.join(BASE_DIR, "src/conf"), config_name="training_config", version_base=None)
def main(cfg):

    # wandb_logger = pl.loggers.WandbLogger(
    #     name = wandb.api.viewer()["username"] + " - " + datetime.now().strftime("%Y-%m-%d %H:%M"), # run name format : username - timestamp 
    #     log_model=False, # if log_model == 'all', checkpoints are logged during training.
    #                     # if True, checkpoints are logged at the end of training, except when save_top_k == -1 which also logs every checkpoint during training.
    #                     # if log_model == False (default), no checkpoint is logged.
    #     project="Classification of dogbreeds",
    #     config={
    #         "architecture": "PretrainedVGG16",
    #         "dataset": "8DogBreeds",
    #         "learning_rate": cfg.lr,
    #         "batch_size" : cfg.batch_size,
    #         "max_epochs": cfg.max_epochs,
    #         "validation_ratio" : cfg.validation_ratio,
    #         "testing_ratio" : cfg.testing_ratio,
    #         "transforms" : "target_shape:"+str(tuple(cfg.target_shape)),
    #         "commit_message": commit_message,
    #         "commit_hash": commit_hash
    #     })

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
    # trainer = pl.Trainer(logger=wandb_logger, log_every_n_steps=1, devices=1, accelerator=accelerator, max_epochs=cfg.max_epochs)
    trainer = pl.Trainer(log_every_n_steps=1, devices=1, accelerator=accelerator, max_epochs=cfg.max_epochs)


    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Test
    loss = trainer.test(model, dataloaders=test_loader)

    test_loss = loss[0]["test_loss"]


    # Update the save model checkpoint if it has a lower loss than the current best model
    if (float(os.environ["MIN_TEST_LOSS"]) > test_loss):
        trainer.save_checkpoint("models/model.ckpt")
        os.environ["MIN_TEST_LOSS"] = str(test_loss)
    
    return test_loss


if __name__ == "__main__":
    main()
    
