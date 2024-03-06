# mlops_final

Final Project in Machine Learning Operations Jan 2024 at DTU

Magnús Sigurðarson s222720
Jónas Már Kristjánsson s223596
Mohammadamin Hassanpour 230883
Alborz Sabet s202075

## Overall goal of the project
The goal of this project was to use PyTorch (with Pytorch image models) to solve the classification task of identifying the dog breed of an image. We will limit ourselves to classifying the following breeds: beagle, bulldog, dalmatian, german-shepherd, husky, labrador-retriever, poodle and rottweiler.

[Launch the Demo](https://dogbreedclassifier.streamlit.app/)

## Frameworks
Since this is an image classification task we will be using pytorch-image-models from huggingface.

## Data
We will be using the Dog Breeds dataset from: https://www.kaggle.com/datasets/mohamedchahed/dog-breeds/data

## Models
We expect to experiment with using pre-trained models, e.g. https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html#torchvision.models.VGG16_Weights while also creating our own from scratch and compare performance.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/Black3rror/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


