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
for batch in data:
    print("data shape: " + str(np.shape(batch)))

def GetRandomSubsetOfData(data, batch_size):
    indices = np.random.randint(0, len(data), size=batch_size)
    return data[indices]

from tensorflow.keras import backend as K
with tf.device("/GPU:0"):
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(24, activation=tf.nn.relu))
    # model.add(Dense(1, activation=tf.nn.sigmoid))
    # model.add(Dropout(0.2))
    # model.add(Dense(24, activation=tf.nn.relu))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(x=data, y=None, epochs=10)
    # model.convert = K.function(inputs=virtual_input, outputs=x)