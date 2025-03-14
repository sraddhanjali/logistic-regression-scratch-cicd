#!/bin/bash

# install environment 
bash tools/install.sh
source .venv/bin/activate
# Runs the build workflow steps for testing purposes
# Exit when any command fails

set PWD /Users/sam/porfolio_workspace/Logistic-Regression-from-Scratch
set -e

# TODO lint only config.yaml file
# python3 config.yaml

# features tests
python3 $PWD/tests/test_mlflow.py

set -e

python3 $PWD/tests/test_preprocessing.py

set -e 

python3 $PWD/tests/test_features.py

set -e 

python3 $PWD/tests/test_pipeline.py

set -e

python3 $PWD/tests/test_classifier.py

set -e

# TODO with bite-size operations

# python3 ml/build.py
# python3 ml/train.py
# python3 ml/test.py
echo "Success!"

exec $SHELL