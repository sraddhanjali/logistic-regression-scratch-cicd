# Design a streamlined ML model (Logisitic Regression)
## Features include:
1. Dockerization compatibility
2. Proper testing
3. Modularization
4. Offline training support

### Project Structure


ml_project/

│── data/                   - Data folder (bind-mounted in Docker)

│── models/                 - Trained models saved here

│── tests/                  - Unit tests

│   ├── test_utils.py       

│   ├── test_ml_model.py  

│── utils/                  

│   ├── utils.py            - General utility functions

│   ├── preprocessing.py    - Preprocessing functions

│── ml_model.py             - Main model training script

│── train.py                - Training pipeline script

│── mlflow_tracking.py      - MLflow integration

│── Dockerfile              - Docker setup

│── requirements.txt        - Dependencies

│── run_offline.py          - Offline testing scenario

│── README.md               - Documentation

