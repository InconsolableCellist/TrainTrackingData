import pickle, os
from random import random

import numpy as np
import torch
import time
import wandb

from Bottleneck import Bottleneck
from VRCDataset import VRCDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE      = "cuda:0"

os.environ["WANDB_MODE"] = "online"
TRAIN       = True
SAVE_MODEL  = True
LOAD_CHECKPOINT = False
CHECKPOINT = os.path.join('models', 'blackcatmodel_no_normalization-steps_100-batchsize_5000-epochs_50-latentsize_2048')

DATASET_FILE = 'blackcatlocalposition_no_normalization.pkl'
DATASET_PATH = 'dataset'
MODEL_NAME = 'blackcatmodel_no_normalization'

NUM_PLAYERS = 30
NUM_DATA_POINTS = 42  # (3 floats for 14 tracked items)

dataset = {}
with open(os.path.join(DATASET_PATH, DATASET_FILE), 'rb') as f:
    input = pickle.load(f)

data    = np.asarray(input['data'], dtype=object)
worldUUID    = input['worldUUID']
sessionStart = input['sessionStart']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sequence_size   = 32 # ~3 per second
# num_epochs      = 75
num_epochs      = 50
batch_size      = 5000
steps_per_epoch = 100
latent_size     = 3072
seq_index       = 0
ses_index       = 0

data_attention = np.ones(data.shape)

def save_data_attention(data_attention, filename):
    print(f'Saving data_attention to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(data_attention, f)

zeros = np.zeros_like(data)
ones  = np.ones_like(data)
# data_attention = np.where(np.all(data > 0, axis=-1, keepdims=True), ones, zeros)

test_data  = data[-1:]
train_data = data[:-1]

# test_attention  = data_attention[-1:]
# train_attention = data_attention[:-1]


def read_one(data):
    global seq_index, ses_index
    sample = torch.tensor(data[ses_index][seq_index:seq_index+sequence_size])
    seq_index += sequence_size
    if seq_index + sequence_size >= data[ses_index].shape[1]:
        seq_index = int(random() * sequence_size)
        ses_index += 1
        if ses_index >= data.shape[0]:
            ses_index = 0
    return sample

def read_batch(data):
    return torch.stack([ read_one(data) for e in range(batch_size) ], dim=0)

print(f'data shape: {data.shape}')
print(f'training data shape: {train_data.shape}')
print(f'test data shape: {test_data.shape}')
# print(f'read_one shape: {read_one(train_data).shape}')
# print(f'read_batch shape: {read_batch(train_data).shape}')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def pytorch_train_on_data(model, optimizer, loss_fn, epochs, steps_per_epoch):
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for step in range(steps_per_epoch):
            batch = read_batch(train_data)
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            batch_flat = torch.reshape(batch, (batch.shape[0], batch.shape[1], batch.shape[2] * batch.shape[3]))
            # print(f'\tbatch_flat shape: {batch_flat.shape}')
            # ([9, 2048, 2400])

            output = model(batch_flat[:, :-1])

            output = torch.reshape(output, (batch.shape[0], 1, NUM_PLAYERS, NUM_DATA_POINTS))
            target = torch.reshape(batch[:, -1], (batch.shape[0], 1, NUM_PLAYERS, NUM_DATA_POINTS))
            loss = loss_fn(output, target)
            # if not step % 10:
            wandb.log({'loss': float(loss)})
            print(f'\tStep {step} loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        eval(model, loss_fn)

def eval(model, loss_fn):
    print(f'Evaluating model')
    with torch.no_grad():
        batch = read_batch(train_data)
        batch = batch.to(DEVICE)

        batch_flat = torch.reshape(batch, (batch.shape[0], batch.shape[1], batch.shape[2] * batch.shape[3]))
        output = model(batch_flat[:, :-1])
        output = torch.reshape(output, (batch.shape[0], 1, NUM_PLAYERS, NUM_DATA_POINTS))

        # loss = loss_fn(output[:, :-1], batch[:, 1:])
        target = torch.reshape(batch[:, -1], (batch.shape[0], 1, NUM_PLAYERS, NUM_DATA_POINTS))
        loss = loss_fn(output, target)

        wandb.log({'eval_loss': float(loss)})
        print(f'\tTraining loss: {loss.item()}')

def pytorch_define_lstm_model(latent_size):
    model = torch.nn.Sequential(
        Bottleneck(latent_size=latent_size, num_players=NUM_PLAYERS, num_datapoints=NUM_DATA_POINTS),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size, latent_size),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size, latent_size//2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size//2, latent_size//4),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(latent_size//4, 1*NUM_PLAYERS*NUM_DATA_POINTS),
    )
    model.to(DEVICE)
    return model

def prepare_embedding():
    embedding = torch.nn.Embedding(num_embeddings=NUM_PLAYERS, embedding_dim=1)
    for player in range(NUM_PLAYERS):
        embedding.weight.data[player, 0] = torch.tensor([player])
    return embedding

def save_model_pkl(model, model_filename):
    if os.path.exists(model_filename + '.pkl'):
        model_filename = model_filename + '-' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
    else:
        model_filename = model_filename + '.pkl'
    print(f'Saving model to {model_filename}')
    pickle.dump(model, open(model_filename, 'wb'))

def load_model_pkl(model_filename):
    print(f'Loading model from {model_filename}')
    model = None
    if os.path.exists(model_filename  + '.pkl'):
        model = pickle.load(open(model_filename + '.pkl', 'rb'))
    return model

wandb.init(name="VRCTrackingModel", project="VRCTrackingModel")

model       = pytorch_define_lstm_model(latent_size=latent_size)
optimizer   = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn     = torch.nn.MSELoss()
embedding   = prepare_embedding() #torch.nn.Embedding(num_embeddings=NUM_PLAYERS, embedding_dim=1)

if LOAD_CHECKPOINT is True:
    model_meta = load_model_pkl(CHECKPOINT)
    model = model_meta['model']
    print(f'Loaded model from checkpoint')

print(f'Steps per epoch: {steps_per_epoch}')
print(f'Batch size: {batch_size}')
print(f'Training on {num_epochs} epochs')
print(f'Device: {DEVICE}')
print(f'Loss_fn: {loss_fn}')
if TRAIN == True:
    pytorch_train_on_data(model=model, optimizer=optimizer, loss_fn=loss_fn, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

output = { 'model': model,
           'worldUUID': worldUUID,
           'sessionStart': sessionStart }

filename = os.path.join('models', f'{MODEL_NAME}-steps_{steps_per_epoch}-batchsize_{batch_size}-epochs_{num_epochs}-latentsize_{latent_size}')
if SAVE_MODEL is True:
    save_model_pkl(output, filename)

print(f'Loading model from {filename} to test it was saved properly')
model_meta  = load_model_pkl(filename)
model       = model_meta['model']
eval(model, loss_fn)