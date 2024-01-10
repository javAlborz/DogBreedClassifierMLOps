import pytest
import torch
import torchvision.transforms as transforms
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.train_model import main

@patch('wandb.api')
@patch('pytorch_lightning.loggers.WandbLogger')
def test_main(mock_wandb_logger, mock_wandb_api):
    # Define the configuration parameters
    class Config:
        lr = 0.001
        batch_size = 32
        max_epochs = 10
        validation_ratio = 0.2
        testing_ratio = 0.1
        target_shape = (224, 224)

    # Create a mock configuration object
    cfg = Config()

    # Create a mock viewer object with a username attribute
    mock_viewer = MagicMock()
    mock_viewer.__getitem__.return_value = 'test_username'
    # Set the return value of wandb.api.viewer() to the mock viewer
    mock_wandb_api.viewer.return_value = mock_viewer

    # Create a mock WandbLogger object
    mock_logger = MagicMock()

    # Set the return value of pl.loggers.WandbLogger() to the mock logger
    mock_wandb_logger.return_value = mock_logger


    # Call the main function
    model, trainer = main(cfg)

    # Assert that the model parameters are trainable
    assert all(p.requires_grad for p in model.parameters()), "Expected all model parameters to be trainable"

    # Assert that the model is in training mode
    assert model.training, "Expected model to be in training mode"

    # TODO: Assert that the checkpoints and model files are saved correctly


    print("All tests passed!")