FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY gsutil -m cp -r gs://mlops-group13-models/model.ckpt models/

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./src /code/app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]