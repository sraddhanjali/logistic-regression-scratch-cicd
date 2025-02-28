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
from utils.config import CONFIG

script_dir = sys.path[0]

# Loading a synthetic training data
print(f"Loading the training data")
