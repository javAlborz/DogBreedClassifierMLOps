# Base image
FROM --platform=linux python:3.10-slim

RUN apt update
RUN apt install --no-install-recommends -y build-essential gcc
RUN apt install -y git 
RUN apt clean 
RUN rm -rf /var/lib/apt/lists/*


WORKDIR /DogBreedClassifierMLOps
COPY src/ src/
# COPY data/ data/ #dont iclude data in docker image. DVC will make it
#COPY reports/ reports/
COPY models/ models/

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install dvc-gs
RUN pip install -r requirements.txt  --no-cache-dir

#COPY .git .git
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY data/raw.dvc data.dvc
RUN dvc config core.no_scm true

#RUN git init
#COPY .dvc .dvc
#COPY data.dvc data.dvc
#RUN dvc pull

CMD ["sh", "-c", "dvc pull && python src/data/make_dataset.py && python src/train_model.py"]
