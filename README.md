# mlops_final

Final Project in Machine Learning Operations Jan 2024 at DTU

Magnús Sigurðarson s222720
Jónas Már Kristjánsson s223596
Mohammadamin Hassanpour 230883

## Overall goal of the project
The goal of this project is to use PyTorch (with Pytorch image models) to solve the classification task of identifying the dog breed of an image. We will limit ourselves to classifying the following breeds: beagle, bulldog, dalmatian, german-shepherd, husky, labrador-retriever, poodle and rottweiler.

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



## TODO


### Week 1
- [X] Create a git repository
- [X] Make sure that all team members have write access to the github repository
- [ ] Create a dedicated environment for you project to keep track of your packages
- [X] Create the initial file structure using cookiecutter
- [X] Fill out the make_dataset.py file such that it downloads whatever data you need and
- [X] Add a model file and a training script and get that running
- [ ] Remember to fill out the requirements.txt file with whatever dependencies that you are using
- [ ] Remember to comply with good coding practices (pep8) while doing the project
- [ ] Do a bit of code typing and remember to document essential parts of your code
- [ ] Setup version control for your data or part of your data
- [ ] Construct one or multiple docker files for your code
- [ ] Build the docker files locally and make sure they work as intended
- [ ] Write one or multiple configurations files for your experiments
- [ ] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally, consider running a hyperparameter optimization sweep.
- [X] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code




### Week 2
- [ ] Write unit tests related to the data part of your code
- [ ] Write unit tests related to model construction and or model training
- [ ] Calculate the coverage.
- [ ] Get some continuous integration running on the github repository
- [ ] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
- [ ] Create a trigger workflow for automatically building your docker images
- [ ] Get your model training in GCP using either the Engine or Vertex AI
- [ ] Create a FastAPI application that can do inference using your model
- [ ] If applicable, consider deploying the model locally using torchserve
- [ ] Deploy your model in GCP using either Functions or Run as the backend


### Week 3
- [ ] Check how robust your model is towards data drifting
- [ ] Setup monitoring for the system telemetry of your deployed model
- [ ] Setup monitoring for the performance of your deployed model
- [ ] If applicable, play around with distributed data loading
- [ ] If applicable, play around with distributed model training
- [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed


### Additional
- [ ] Revisit your initial project description. Did the project turn out as you wanted?
- [ ] Make sure all group members have a understanding about all parts of the project
- [ ] Uploaded all your code to github






