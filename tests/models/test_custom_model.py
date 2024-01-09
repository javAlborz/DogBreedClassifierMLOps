import pytest
import torch

from src.models.custom_model import CustomCNN


@pytest.mark.parametrize("input_shape", [(1, 28, 28), (3, 32, 32)])
def test_custom_cnn(input_shape):
    # Define the model parameters
    output_size = 10
    convs_params = [(32, 3, 1), (0, 2, 2), (64, 3, 1)]
    denses_params = [128, 64]
    conv_dropout = 0.2
    dense_dropout = 0.5
    random_seed = 42

    # Create an instance of the model
    model = CustomCNN(input_shape, output_size, convs_params, denses_params, conv_dropout, dense_dropout, random_seed)

    # Generate random input data
    batch_size = 16
    input_data = torch.randn(batch_size, *input_shape)

    # Perform a forward pass
    output = model(input_data)

    # Assert the output shape
    assert output.shape == (batch_size, output_size), "Expected output shape of {}, but got {} instead".format((batch_size, output_size), output.shape)

    # Assert that the output probabilities sum up to 1
    assert torch.allclose(torch.exp(output).sum(dim=1), torch.ones(batch_size)), "Expected output probabilities to sum up to 1"

    # Assert that the model parameters are trainable
    assert all(p.requires_grad for p in model.parameters()), "Expected all model parameters to be trainable"

    # Assert that the model is in training mode
    assert model.training, "Expected model to be in training mode"

    print("All tests passed!")