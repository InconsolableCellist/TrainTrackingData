import pickle, os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
device      = "cuda:0"

DATASET_FILE = 'blackcatlocalposition.pkl'
DATASET_PATH = 'dataset'

dataset = {}
with open(os.path.join(DATASET_PATH, DATASET_FILE), 'rb') as f:
    data = pickle.load(f)

print("data shape: " + str(np.shape(data)))


