import pytest
import os
import glob
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

from src import data_loader

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","data")
Number_of_images = len(glob.glob(os.path.join(BASE_DIR, "raw","*","*","*.jpg")))

def test_classname_to_number_map():
    classname_to_number_map = {
        "beagle" : 0,
        "bulldog" : 1,
        "dalmatian" : 2,
        "german-shepherd" : 3,
        "husky" : 4,
        "labrador-retriever" : 5,
        "poodle" : 6,
        "rottweiler" : 7,
    }

    assert data_loader.classname_to_number_map == classname_to_number_map


@pytest.mark.parametrize("batch_size,validation_ratio,testing_ratio,target_size", 
                         [(10, 0.2, 0.1,(128,128)), 
                          (30, 0.3, 0.2,(128,128)),
                          (100, 0.5, 0.1,(64,64))])
def test_data_loader(batch_size,validation_ratio,testing_ratio,target_size):
    train_loader, valid_loader, test_loader = data_loader.get_data(batch_size=batch_size, validation_ratio=validation_ratio, testing_ratio=testing_ratio, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(target_size, antialias=True)]))

    # Check type 
    assert type(train_loader) == DataLoader   
    assert type(valid_loader) == DataLoader   
    assert type(test_loader) == DataLoader   

    # Number of data in each loader
    assert len(train_loader.dataset) == int(Number_of_images * (1.0 - validation_ratio - testing_ratio))
    assert len(valid_loader.dataset) == int(Number_of_images * validation_ratio)
    assert len(test_loader.dataset) == int(np.ceil(Number_of_images * testing_ratio))

    # Batch size
    assert train_loader.batch_size == batch_size
    assert valid_loader.batch_size == batch_size
    assert test_loader.batch_size == 1

    # Number of batches
    assert len(train_loader) == np.ceil(len(train_loader.dataset)/batch_size)
    assert len(valid_loader) == np.ceil(len(valid_loader.dataset)/batch_size)
    assert len(test_loader) == len(test_loader.dataset)

    # Iterators
    batch_counter = 0
    for X,Y in train_loader:
        batch_counter+=1
        if batch_counter == len(train_loader): # Skip the last since they are not the same size
            continue
        assert X.shape == torch.Size([batch_size, 3, target_size[0], target_size[1]])
        assert torch.min(Y).item() >= 0
        assert torch.max(Y).item() <= len(data_loader.classname_to_number_map)

        
        



    
    


