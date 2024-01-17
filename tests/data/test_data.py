import glob
import os
import re

import numpy as np
from PIL import Image

from src import data_loader

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","data")

def get_class_name(path):
    filename = os.path.basename(path)
    index_first_number = re.search(r'\d+', filename).start()
    return filename[:index_first_number]

def count_classes(list_of_paths):
    counting_dict = dict.fromkeys(data_loader.classname_to_number_map.keys(), 0)
    for path in list_of_paths:
        counting_dict[get_class_name(path)] += 1

    return counting_dict   
        

def test_data_sizes():
    list_of_processed = glob.glob(os.path.join(BASE_DIR, "processed","*.jpg"))
    list_of_raw = glob.glob(os.path.join(BASE_DIR, "raw","*","*","*.jpg"))
    assert len(list_of_processed) == len(list_of_raw)

    class_count_in_processed = count_classes(list_of_processed)
    class_count_in_raw = count_classes(list_of_raw)
    for class_name in class_count_in_processed.keys():
        assert class_count_in_processed[class_name] == class_count_in_raw[class_name]


def test_image_size_and_values():
    img = Image.open(os.path.join(BASE_DIR, "processed","beagle1.jpg"))
    assert img.getbands() == ("R","G","B")
    assert img.width == 509
    assert img.height == 339
    assert np.all(np.sum(np.sum(np.array(img),axis=0),axis=0) == [22650398, 20419741, 18473145]) 










