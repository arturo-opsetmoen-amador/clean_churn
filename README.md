# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity. 

## Project Description
Implementation of production code practices such as testing and logging. The 
project takes as a starting point the analysis and modelling contained in the
``` churn_notebook.ipynb```. The project includes a ```test``` folder with 
configurations, initializations, fixtures all used as part of the pytest testing
suite. 

To improve reproducibility, we provide a docker container for setting up the
environment with all dependencies (```requirements.txt```) necessary to run 
the churn analysis.

## Running Files

To run the analysis 

1. Clone the repository: 

```git clone https://github.com/digitalemerge/clean_churn```

2. Change to the directory containing the cloned code (```clean_churn``` by default)
3. Run 

```docker build -t udacity/churn .```

to build the docker container from the Dockerfile in the repository and tag it.
This might take a while. 

4. Run 

```docker run -v $(pwd):/clean_churn udacity/churn```

This will "hit" the entrypoint command: ```run.sh```. This script executes:

- The ```churn_script_logging_and_tests.py``` test inside the tests folder (pytest)
- The ```churn_library.py``` pipeline to perform the churn analysis. 
- Two linter analysis (pylint), one for ```churn_script_logging_and_tests.py```
and one for ```churn_library.py```

As a by product two folders will be created: 

- artifacts. This contains two sub-folders: ```images``` and ```models```. 
The ```images``` folder contains the results of the eda (under the ```eda``` 
folder) and the results after modelling (in the ```results``` folder)
- logs. This directory contains the logs created by pytest after performing the 
tests. In addition this folder contains the reports from ```pylint``` for
```churn_script_logging_and_tests.py ``` and ```churn_library.py ```


GL, HF!
