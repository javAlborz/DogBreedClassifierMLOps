FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
# RUN gsutil -m cp -r gs://mlops-group13-models/model.ckpt models/

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY models/ models/
COPY src/ src/

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]     

#You can start server with:
# docker run -p 8000:80 apitest:latest
#Then run:
# curl -X POST -F "data=@path_to_your_image/dog.jpg" http://localhost:8000/model
