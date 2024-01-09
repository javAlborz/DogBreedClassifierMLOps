""" Current model -- VGG16 pretrained model """

import torch
import torchvision
import lightning as L

class MyNeuralNet(L.LightningModule):
    """ Basic neural network class. """

    def __init__(self,lr) -> None:
        """Initialize 

        Args:
            lr (float) : Learning rate
        """
        super().__init__()

        self.lr = lr

        # Load in pretrained VGG16 model and replace the last layer in the classifier with the 
        # number of dog-breed classes
        self.model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT) 
        self.model.classifier[-1] = torch.nn.Linear(4096,8, bias=True) 

        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model. 
        
        Args:
            x (torch.Tensor) : Batch of images (batch_size, channels, height, width)

        Returns:
            (torch.Tensor) : logits results from the model (batch_size, 8 i.e. number of output classes) 
        
        """
        return self.model(x)
    
    def training_step(self, batch : tuple, batch_idx : int) -> torch.Tensor:
        """"Training step on the lightning format

        Args:
            batch (tuple) : Contains the batch data i.e. batch of images and labels
            batch_idx (int) : index of the batch

        Returns: 
            (torch.Tensor) : Loss for batch
        """
        x, targets = batch
        preds_logits = self.forward(x)
        loss = self.loss_func(preds_logits, targets)
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch : tuple, batch_idx : int) -> torch.Tensor:
        """"Validation step on the lightning format

        Args:
            batch (tuple) : Contains the batch data i.e. batch of images and labels
            batch_idx (int) : index of the batch

        Returns: 
            (torch.Tensor) : Loss for batch
        """
        x, targets = batch
        preds_logits = self.forward(x)
        loss = self.loss_func(preds_logits, targets)
        self.log("val_loss", loss)
        return loss
    
    
    def test_step(self, batch : tuple, batch_idx : int) -> torch.Tensor:
        """"Testing step on the lightning format

        Args:
            batch (tuple) : Contains the batch data i.e. batch of images and labels
            batch_idx (int) : index of the batch

        Returns: 
            (torch.Tensor) : Loss for batch
        """
        x, targets = batch
        preds_logits = self.forward(x)
        loss = self.loss_func(preds_logits, targets)
        self.log("test_loss", loss)
        return loss


    def configure_optimizers(self):
        """" Set the optimization parameters

        Returns:
            (torch.nn.Optimizer) : Configure optimizer for model     
        """
        return torch.optim.Adam(self.model.parameters(), lr = self.lr)
    


    

    


    


