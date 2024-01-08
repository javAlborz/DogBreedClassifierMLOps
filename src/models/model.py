import torch
import torchvision

class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. """

    def __init__(self) -> None:
        super().__init__()

        # Load in pretrained VGG16 model and replace the last layer in the classifier with the 
        # number of dog-breed classes
        self.model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT) 
        self.model.classifier[-1] = torch.nn.Linear(4096,8, bias=True) # 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model. """
        return self.model(x)
    


