import torch
from pytorch_lightning import LightningModule
from torch import nn, optim


class CustomCNN(LightningModule):
    """ Custom CNN class."""

    def __init__(self, input_shape, output_size, convs_params, denses_params, conv_dropout, dense_dropout, random_seed=None):
        """Initialize the custom CNN model.

        Args:
            input_shape (tuple): Shape of the input data (C, H, W)
            output_size (int): Number of classes
            convs_params (list): List of tuples, parameters for convolutional layers. Each tuple is (c, k, s) of a conv layer
            denses_params (list): List of tuples, parameters for dense layers. Each element is the number of neurons of a dense layer
            conv_dropout (float): Dropout rate for convolutional layers
            dense_dropout (float): Dropout rate for dense layers
            random_seed (int | None): Random seed. If None, no seed is set
        """

        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.convs_params = convs_params
        self.denses_params = denses_params
        self.conv_dropout = conv_dropout
        self.dense_dropout = dense_dropout
        self.random_seed = random_seed
        self.criterion = torch.nn.NLLLoss()

        self.convs = nn.ModuleList()
        self.denses = nn.ModuleList()
        self._build_model()

    def _build_model(self):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        convs_input_chs = [self.input_shape[0]]
        for i, (c, k, s) in enumerate(self.convs_params):
            if c > 0:
                convs_input_chs.append(c)
            elif k > 0:
                convs_input_chs.append(convs_input_chs[-1])
            else:
                raise NotImplementedError

        division = 1
        for i, (c, k, s) in enumerate(self.convs_params):
            division *= s
        c_last = 0
        i = len(self.convs_params) - 1
        while c_last == 0:
            c_last = self.convs_params[i][0]
            i -= 1
        denses_input_size = [c_last * (self.input_shape[1] // division) * (self.input_shape[2] // division), *self.denses_params[:-1]]

        for i, (c, k, s) in enumerate(self.convs_params):
            if c > 0:
                self.convs.append(nn.Conv2d(in_channels=convs_input_chs[i], out_channels=c, kernel_size=k, stride=s, padding=(k - 1) // 2))
                self.convs.append(nn.BatchNorm2d(c))
                self.convs.append(nn.ReLU())
                self.convs.append(nn.Dropout2d(self.conv_dropout))
            elif k > 0:
                self.convs.append(nn.MaxPool2d(kernel_size=k, stride=s, padding=(k - 1) // 2))
            else:
                raise NotImplementedError
        
        for i, d in enumerate(self.denses_params):
            self.denses.append(nn.Linear(denses_input_size[i], d))
            self.denses.append(nn.BatchNorm1d(d))
            self.denses.append(nn.ReLU())
            self.denses.append(nn.Dropout(self.dense_dropout))
        
        self.denses.append(nn.Linear(self.denses_params[-1], self.output_size))
        self.denses.append(nn.LogSoftmax(dim=1))

    def forward(self, x):
        """
        Forward pass of the custom model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        for layer in self.convs:
            x = layer(x)
        
        x = x.view(x.size(0), -1)

        for layer in self.denses:
            x = layer(x)

        return x
    
    def training_step(self, batch, batch_idx):
        """ Perform a training step.

        Args:
            batch (tuple): Tuple containing the input and target tensors.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Perform a validation step.

        Args:
            batch (tuple): Tuple containing the input and target tensors.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """ Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer