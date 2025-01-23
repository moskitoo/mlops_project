# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [X] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [X] Add a continues workflow that triggers when data changes (M19)
* [X] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X] Create a trigger workflow for automatically building your docker images (M21)
* [X] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X] Create a FastAPI application that can do inference using your model (M22)
* [X] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X] Write API tests for your application and setup continues integration for these (M24)
* [X] Load test your application (M24)
* [X] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [x] Setup cloud monitoring of your instrumented application (M28)
* [x] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

53

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s242954, s243575, s242529, s242672, s243600

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

In this project, we utilized the PyTorch-based Ultralytics package to implement the YOLO model (version 11) for object detection. This package simplifies the process of training and evaluation, reducing boilerplate code similarly to PyTorch Lightning.

We used pretrained YOLO models on the COCO dataset as a base and leveraged Ultralytics' built-in features for metrics logging, report generation, and model exporting. The models were exported in PyTorch (.pth) and ONNX (.onnx) formats for deployment.

The package offers seamless integration with Weights & Biases (W&B), enabling effective tracking of experiments and model versions. Results were stored in organized directories, ensuring reproducibility and easy collaboration.

We initially considered Detectron2 but chose Ultralytics due to its active maintenance, popularity, and ease of debugging. The extensive documentation and community support made it a practical choice for our needs.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We managed our project dependencies using requirements.txt, requirements_dev.txt, and pyproject.toml. The requirements.txt file contains the core dependencies required to run the project, while requirements_dev.txt includes additional packages for development and testing. The pyproject.toml file provides metadata about the project and specifies build system requirements, ensuring compatibility and consistency. 

To replicate the environment, a new team member would need to clone the repository, create and activate a virtual environment, and install the dependencies listed in the requirements.txt and requirements_dev.txt files. Alternatively, they can use the pyproject.toml file to install dependencies in one step. This structured approach ensures that every team member has an identical development environment, reducing issues caused by dependency mismatches.

1. Clone the repository:
```bash
git clone https://github.com/moskitoo/mlops_project.git
cd mlops_project
```
2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install all dependencies using pyproject.toml
```bash
pip install .
```



### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

From the cookiecutter we have filled out the , reports, test (with the pytest unit tests), we stored all of our model’s python scripts under src/pv_defection_classification, in the configs folder we saved the running configurations, in the dockerfiles folder we used to save all different docker configurations. In the data we stored the data files as well as the dataset.yaml configurations. We didn’t use notebooks, and docs. We also added the “preformance_test” for running evaluations of our models not runned by the pytest auto unit tests. 

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

In our project, we used Ruff to manage code quality and formatting. We incorporated ruff check and ruff format as steps in our continuous integration process to ensure compliance with PEP 8. We maintained the basic settings, only increasing the line length to 120, as specified in the exercises. For type checking, we used MyPy. Code documentation, including docstrings and inline comments, was reviewed using GitHub Copilot.

In larger projects, even within a team of our size, proper documentation is crucial, especially when integrating different parts of the project. In our case, collaboration was slightly easier because everyone had completed the exercises and was familiar with the tools used across the project. However, in other scenarios where such prior knowledge might not be available, clear documentation can be extremely helpful.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented 3 tests.

For the data processing component, we created four unit tests and two integration tests. The unit tests focused on handling cases of missing or corrupted data during processing, ensuring that the raw dataset was correctly transformed into the YOLO format. The integration tests verified that the overall structure of the processed dataset met the required format and organization needed for the training step of our pipeline.

For the model, we have implemented three unit tests to validate the loading and saving mechanisms. The unit tests verify that pretrained models are loaded correctly with or without weights and that the saving process ensures models are stored at the correct paths with proper directory handling.

For the API we implemented one performance test using the locust framework and and three unit tests to ensure that the internal functions work as expected. We also implemented one integration test to test whether the API behaves as expected.


### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The model.py file has 100% code coverage, meaning every line of code is tested. This shows that we’ve thoroughly validated its functionality and reliability. While 100% coverage doesn’t guarantee there are no errors, the extensive tests and attention to edge cases make this part of the project highly reliable and trustworthy.

The code coverage for the API is 30%. We do not reach 100% here because the API includes functions that drwa on images where the proper functionality can only be inspected visually. Our tests cover the critical parts of our API, including the initialisation, preprocessing and prediction steps. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In our project, we made use of both branches and PRs. We created new branches for each new feature we wanted to develop. We encountered some issues with commit and PR frequency, but this project taught us how important it is to regularly save modifications and merge them to ensure that everyone stays on the same page. We added branch protection in the later phase of our project when we moved from local development to the cloud. At that stage, our project consisted of many nodes, and an incorrect push to the main branch could trigger a cascade of events in the pipeline, wasting our time. However, we are aware that in more advanced projects, especially in a real production environment, branch protection is crucial from day one. The risks are even higher, for example, the GCP tokens that we pay for with our own money. :D

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC in the early stages as an easy way to access data for development. It allowed us to retrieve the processed dataset stored on GCP with a single command, eliminating the need to process it repeatedly, such as when working on our Docker container.

However, for operations running in the cloud, we decided to access the data directly from GCP storage, which is auto-mounted, such as when running a Vertex AI job. Pulling data using DVC while building a container on Cloud Build turned out to be a significant bottleneck in the pipeline. 

In the later stages of the project, when introducing new features, we used changes in DVC metafiles to trigger a GitHub Action that verifies whether the available datasets have the correct structure after data modifications. It also displays information about the datasets and sample images. Thanks to this, our pipeline offers better data protection and provides an easy way to inspect the data. 

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have organised our continuous integration pipeline into 3 separate files: one for detecting changes in the data, one for running our tests and starting the builds of our containers on GCP and one to detect changes in the model registry. In particular for our workflow for detecting changes in the model registry we used a weights and biases webhook to trigger a workflow in our repository. 

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

In order to run an experiment we should run the train.py file. There are 2 possible configurations for this run: if you run train.py without arguments, the code will identify that (out of the typer.context variable) and use configs/config.yaml otherwise, the code will use the given arguments and default code parameters. So you can go as Python train.py and also python train.py --batch_size 32 learning_rate 0.0025 --max_iteration 100 

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

As stated above, we used both config files and cli, during every experiment the YOLO model configurations are saved in a dictionary that is uploaded to weights and biases. In order to reproduce the results of an experiment, one would need to copy the hyperparameters from WandB to config.yaml file, and just run the experiment as mentioned above. 

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

[Figure 1](figures/data_log.png)
[Figure 2](figures/hyperparameter_sweep.png)
[Figure 3](figures/model_save_artifact.png)

We have used Weights & Biases (W&B) in our project for logging data to visualize metrics, performing hyperparameter sweeps, and saving model artifacts. Metrics like recall and precision helped us ensure the system accurately detected defective photovoltaic modules while minimizing false detections. We tracked mAP50-95 for a comprehensive evaluation of detection accuracy and monitored losses such as distributive focal loss, classification loss, and bounding box loss to ensure effective model learning. The learning rate schedules were analyzed to optimize the training process. Additionally, hyperparameter sweeps allowed us to fine-tune configurations like batch size and learning rate for the best performance. Saving model artifacts in W&B ensured easy access and versioning logging. All of this could be referred in the figures 1, 2 and 3. 

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We developed several docker images for our project. These include images for training on cpu and gpu and docker images for deploying our application. The docker image for our API can be run with the command `docker run bento_service:latest` and starts a bentoml api on `http://localhost:3000`. This API can then be interacted with by using a bentoml client or any other client that can send http requests.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following services: Bucket, Artifact Registry, Cloud Run. Bucket is sed for storing our data and models. We use it to version our datasets and models. The Artifact Registry is used to store our docker images. We use Cloud Run to deploy our API using the corresponding docker image in the Artifact Registry.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[Overview](figures/bucket_2.png)
[What the data looks like](figures/bucket_1.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[Our Artifact Regsitry](figures/artifact_registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[Part 1](figures/cloud_history_1.png)
[Part 2](figures/cloud_history_2.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We did manage to write an API for our model. We started out with FastAPI but transitioned relatively quickly to BentoML as our application is focused on computer vision. Since BentoML provides native support for numpy arrays we decided that it is the more suitable framework for our task. We intergrated the API with access to our GCP buckets such that we can automatically load the newest model upon deployment. We also export the model to onnx and optimise the computation graph to make inference more efficient and quicker. We also experimented with dnamic batching. However, this did not work well with our application which is why we removed it.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We managed to deploy our API both locally and in the cloud. To achieve this we developed a custom docker container for the API. BentoML can buid these automatically, however, we decided to build a custom one for our application. Running this docker container enable local as well as in the cloud. To serve the model in the cloud we use Cloud Run. This mean that our model only runs when a request to the API is made. To invoke the service a user would call `curl -X POST -F "input=@file.json"<weburl>` where the json should contain a thermography image.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We implemented unit, integration and performance testing. For performance testing we used locust.
For the unit testing we used pytest. We made use of pytest fixtures as well as mocking certain functions to test the functionality of the API. For the intergation testing we used subprocess to start our API in a subprocess and consequently send a request to it and validate its response. 
The performance testing revealed the following results:


### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
