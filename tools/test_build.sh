#!/bin/bash

# Runs the build workflow steps for testing purposes

# Exit when any command fails
set -e

python3 ../ml/build.py
# python3 ../ml/train.py
# python3 ../ml/test.py
echo "Success!"