#!/bin/bash

# Runs the build workflow steps for testing purposes

# Exit when any command fails
set -e

# TODO lint only config.yaml file
# python3 config.yaml

python3 ../ml/build.py
python3 ../ml/train.py
python3 ../ml/test.py
echo "Success!"
