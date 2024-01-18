from google.cloud import storage

storage_client = storage.Client()
model_file = "models/model.ckpt"
bucket_name = "mlops-group13-models"

bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(model_file)
blob.upload_from_filename(model_file)