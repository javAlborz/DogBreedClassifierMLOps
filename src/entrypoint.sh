#!/bin/bash

# Run gsutil command to copy data at runtime
gsutil -m cp -r gs://mlops-group13-dog-breeds/data .

# Run your desired commands
python src/data/make_dataset.py
python src/train_model.py
python src/upload_file.py
