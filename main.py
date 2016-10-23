#%matplotlib inline

# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
      pickle_data = pickle.load(f)
      train_features = pickle_data['train_dataset']
      train_labels = pickle_data['train_labels']
      valid_features = pickle_data['valid_dataset']
      valid_labels = pickle_data['valid_labels']
      test_features = pickle_data['test_dataset']
      test_labels = pickle_data['test_labels']
      del pickle_data  # Free up memoy

      print('Data and modules loaded.')
