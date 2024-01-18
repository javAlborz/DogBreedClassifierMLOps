FROM --platform=linux python:3.10-slim

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


#COPY models/ models/
COPY src/ src/
RUN mkdir models
RUN python src/download_model_file.py

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]     

#You can start server with:
# docker run -p 8000:80 apitest:latest
#Then run:
# curl -X POST -F "data=@path_to_your_image/dog.jpg" http://localhost:8000/model
