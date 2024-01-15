import os
import numpy as np
import glob
import PIL.Image as Image
import re

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


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
data_dir = os.path.join(__file__,"..","..","data",)

class DogDataSet(torch.utils.data.Dataset):
    def __init__(self, split, validation_ratio, testing_ratio, transform=transforms.ToTensor(),random_seed=2):
        self.transform = transform
        self.split = split

        self.image_paths = np.array(sorted(glob.glob(os.path.join(data_dir,"processed","*.jpg"))))

        N = len(self.image_paths)

        self.split = split
        np.random.seed(random_seed)
        indexes = np.arange(N, dtype=int)
        np.random.shuffle(indexes)
        if self.split=="train":
            # first x percentage
            i_start = 0
            i_stop = int(N*(1-validation_ratio-testing_ratio))

        elif self.split=="valid":
            # between train and test percentage 
            i_start = int(N*(1-validation_ratio-testing_ratio))
            i_stop = i_start + int(N*validation_ratio)

        elif self.split=="test":
            # after train + validation percentage
            i_start = int(N*(1-validation_ratio-testing_ratio)) + int(N*validation_ratio)
            i_stop = N
        
        else:
            raise SyntaxError("split is only [\"train\",\"valid\",\"test\"]")
                
        self.image_paths = self.image_paths[indexes[i_start:i_stop]]
        

    def __len__(self):
        'Returns the total number of samples'
        return self.image_paths.size
    

    def __getitem__(self, idx):
        'Generates one sample of data'
        X = Image.open(self.image_paths[idx])
        X = self.transform(X)

        # classname -> id
        filename = os.path.basename(self.image_paths[idx])
        index_first_number = re.search(r'\d+', filename).start()
        Y = classname_to_number_map[filename[:index_first_number]]
        
        return X, Y
        
    
    
            
def get_data(batch_size, validation_ratio, testing_ratio, transform=transforms.ToTensor()) -> tuple[DataLoader, DataLoader, DataLoader]:   

    trainset = DogDataSet(split="train", validation_ratio=validation_ratio, testing_ratio=testing_ratio, transform=transform, random_seed=2)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=3, persistent_workers=True)
    validset = DogDataSet(split="valid", validation_ratio=validation_ratio, testing_ratio=testing_ratio, transform=transform, random_seed=2)
    valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=3,persistent_workers=True)
    testset = DogDataSet(split="test", validation_ratio=validation_ratio, testing_ratio=testing_ratio, transform=transform, random_seed=2)
    test_loader = DataLoader(testset, batch_size=1, num_workers=3,persistent_workers=True)

    
    print('Loaded %d training images' % len(trainset))
    print('Loaded %d validation images' % len(validset))
    print('Loaded %d test images' % len(testset))

    return train_loader, valid_loader, test_loader




# if __name__ == "__main__":

#     # How to use 
#     train_loader, valid_loader, test_loader = get_data(batch_size=5, validation_ratio=0.15, testing_ratio=0.05, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((128,128))]))
#     for x_batch, y_label in train_loader:
#         pass




