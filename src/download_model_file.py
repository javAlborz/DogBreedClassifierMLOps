from google.cloud import storage
import sys
import io
import os

file_obj = io.BytesIO()


storage_client = storage.Client()
model_file = "models/model.ckpt"
bucket_name = "mlops-group13-models"

bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(model_file)

print("Blob exists") if blob.exists() else ("Blob does not exist")

blob.download_to_file(file_obj)

file_obj.seek(0)  # Go back to the start of the file

local_file_path = 'models/model.ckpt'

with open(local_file_path, 'wb') as local_file:
    local_file.write(file_obj.read())
    print(f"Downloaded blob to local file {local_file_path}.")
