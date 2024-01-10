import pytest
import torch
import torchvision
import pytorch_lightning as pl

from src.models.model import MyNeuralNet


@pytest.fixture
def model():
    return MyNeuralNet(lr=0.001)


def test_forward_pass(model):
    input_shape = (3, 224, 224)
    input_data = torch.randn(16, *input_shape)
    output = model.forward(input_data)
    assert output.shape == (16, 8), "Expected output shape of (16, 8), but got {} instead".format(output.shape)


def test_training_step(model):
    batch_size = 16
    input_shape = (3, 224, 224)
    input_data = torch.randn(batch_size, *input_shape)
    targets = torch.randint(0, 8, (batch_size,))
    loss = model.training_step((input_data, targets), 0)
    assert isinstance(loss, torch.Tensor), "Expected loss to be a torch.Tensor"


def test_validation_step(model):
    batch_size = 16
    input_shape = (3, 224, 224)
    input_data = torch.randn(batch_size, *input_shape)
    targets = torch.randint(0, 8, (batch_size,))
    loss = model.validation_step((input_data, targets), 0)
    assert isinstance(loss, torch.Tensor), "Expected loss to be a torch.Tensor"


def test_test_step(model):
    batch_size = 16
    input_shape = (3, 224, 224)
    input_data = torch.randn(batch_size, *input_shape)
    targets = torch.randint(0, 8, (batch_size,))
    loss = model.test_step((input_data, targets), 0)
    assert isinstance(loss, torch.Tensor), "Expected loss to be a torch.Tensor"


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer), "Expected optimizer to be a torch.optim.Optimizer"