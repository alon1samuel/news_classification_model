# 20 Newsgroups Classification Model

This project is for classification of 20 newsgroup documents in these fields:

- medicine
- space
- cryptography
- electronics

Code can be found to classify these documents with logistic regression model or random forest model.


## Out of the box docker -
Needs docker on computer
1. Go to project folder.
2. Build docker image - docker build -t clf_model .
3. Run docker image - docker run -p 8888:8888 clf_model
4. Click on jupyter notebook
5. Run notebook from folder


## Dependencies 
virtualenv
python>=3.6
## Install instruction on Linux
virtualenv -p /path/to/python3 venv_classification_model
source venv_classification_model/bin/activate
pip install -r requirements.txt


## Run commands
To run logistic regression benchmark - 
python ./src/benchmark_lr.py

To run random forest classifier - 
python ./src/sklearn_random_forest.py

## Jupyter notebooks - 
Run - 
Jupyter notebook
Go inside notebooks folder and choose a notebook to run
Press - Run all cells

