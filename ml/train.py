import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from core import run_test_old_way


script_dir = sys.path[0]

# Load the data from the build step
with open(os.path.join(script_dir, 'training_data/train_data.pkl'), 'rb') as f:
    train_images = pickle.load(f)

with open(os.path.join(script_dir, 'training_data/train_labels.pkl'), 'rb') as f:
    train_labels = pickle.load(f)


# Load the data from the build step
with open(os.path.join(script_dir, 'training_data/iris_train_data.pkl'), 'rb') as f:
    train_images = pickle.load(f)

with open(os.path.join(script_dir, 'training_data/iris_train_labels.pkl'), 'rb') as f:
    train_labels = pickle.load(f)


run_test_old_way([train_images, train_labels])