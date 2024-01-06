import glob
import os 
import tqdm
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change working directory to this one

if __name__ == '__main__':
    # Get the data and process it
    list_of_raw_images = glob.glob(os.path.join("..","..","data","raw","*","*","*.jpg"))

    for filename in tqdm.tqdm(list_of_raw_images):
        img = Image.open(filename)
        img.save(os.path.join("..","..","data","processed",os.path.basename(filename)))

