# pv_defection_classification

Final project for MLOps course at DTU.

## Overall goal 

 The overall goal of this project is to develop a computer vision-based system that is able to detect photovoltaic modules and classify them into non-defective and defective. In order to perform the classification into defective and non-defective we rely on thermography images. Both large and small photovoltaic systems are susceptible to failures in their equipment, especially in modules due to operational stresses that are exposed and errors during the installation process of these devices. Although numerous internal and external factors originate these failures, the common phenomenon presented by several of them is hot spots on module defective area. 

The final outcome of this project should be a system that can be interacted with by an end-user. More specifically, the end-user should be able to upload a thermography image and receive an overview of the detected cells and which of these might be defective. Hence, our system would allow for partly automated predictive maintenance of photovoltaic cells. 

### What framework are we going to use 

 Primary Framework: We plan on using Detectron2 to access a variety of pre-trained object detection and segmentation models and apply them to our problem.  

Image augmentation, To perform image augmentation, we plan on using Albumentations. Main Augmentations are geometric transformations, flipping, rotation, brightness adjustment, etc.  

Logging and Tracking: We plan on integrating Weights&Biases for tracking our model training process. W&B will be used for hyperparameter optimization and visualizations to analyze the training process. Additionally, Optuna can be integrated alongside W&B to perform advanced hyperparameter optimization via its sampling and pruning algorithms.  

  

### What data are we going to use 

 We are going to use a dataset provided on Kaggle. The dataset is well documented and include a notebook that demonstrates how to load and interact with the data. 

https://www.kaggle.com/datasets/marcosgabriel/photovoltaic-system-thermography 

The dataset includes two sub-datasets, each containing IR images of PV modules/arrays. The total number of images in the dataset is 137 but each image can contains up to dozens of modules. Each picture has annotations about modules’ BB and classification in a standard .json format. 

### What models are we going to use 

We would like to explore different pertained models for this purpose such as YOLO, Segment Anything (SAM), or Faster R-CNN pretrained on the COCO dataset. 

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
