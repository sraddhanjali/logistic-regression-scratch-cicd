#!/bin/bash

# install environment 
bash tools/install.sh
source .venv/bin/activate
# Runs the build workflow steps for testing purposes
# Exit when any command fails
set -e

# TODO lint only config.yaml file
# python3 config.yaml

python3 ml/test_mlflow.py

set -e

python3 ml/tests/test_preprocessing.py

set -e 

python3 ml/tests/test_features.py

set -e 

python3 ml/tests/test_pipeline.py

set -e


# TODO with bite-size operations

# python3 ml/build.py
# python3 ml/train.py
# python3 ml/test.py
echo "Success!"

exec $SHELL