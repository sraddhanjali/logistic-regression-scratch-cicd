# Design a dockerized custom ML model 
### Logisitic Regression created from scratch is used as an example

[![local makefile test](https://github.com/sraddhanjali/logistic-regression-scratch-cicd/actions/workflows/makefile_test_pipeline.yml/badge.svg)](https://github.com/sraddhanjali/logistic-regression-scratch-cicd/actions/workflows/makefile_test_pipeline.yml)
[![docker build/test/close](https://github.com/sraddhanjali/logistic-regression-scratch-cicd/actions/workflows/docker_test_pipeline.yml/badge.svg)](https://github.com/sraddhanjali/logistic-regression-scratch-cicd/actions/workflows/docker_test_pipeline.yml)

## Features include:
1. Dockerization compatibility
2. Proper testing
3. Modularization
4. Offline training support

### Project Structure


    ml_project/

        │── data/                   Data folder (bind-mounted in Docker)
            
            │── processed/
            
                │── testing_data/
                
                │── training_data/

        │── models/                 Trained models saved here
        
            │── offline_model.pkl
            
            │── saved_model.pkl

        │── tests/                  Unit tests

           │──  test_classifier.py       

           │──  test_ml_datapipeline.py 

           │──  test_optimizerandloss.py 

           │──  test_pipeline.py 

           │──  test_preprocessing.py 

           │──  test_utils.py 

        │── tools/
        
            │── install.sh
            
            │── test_build.sh

        │── utils/                  

           ├── utils.py            General utility functions

           ├── preprocessing.py    Preprocessing functions
        
        │── docker-compose.yml

        │── config.yml

        │── ml_model.py             Main model training script

        │── ml_datapipeline.py      Make dataset according to config yml.

        │── Dockerfile              Docker setup

        │── requirements.txt        Dependencies

        │── README.md               Documentation


