import pickle, os
from random import random

import numpy as np
import torch
import absl
import wandb
from torch.utils.data import DataLoader

from Bottleneck import Bottleneck
from VRCDataset import VRCDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "offline"
DEVICE      = "cuda:0"

DATASET_FILE = 'blackcatlocalposition.pkl'
DATASET_PATH = 'dataset'
MODEL_NAME = 'blackcatmodel'

NUM_PLAYERS = 30

dataset = {}
with open(os.path.join(DATASET_PATH, DATASET_FILE), 'rb') as f:
    input = pickle.load(f)

data    = input['data']
offsets = input['offsets']
worldUUID    = input['worldUUID']
sessionStart = input['sessionStart']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sequence_size   = 32 # ~3 per second
num_epochs      = 50
batch_size      = 40
steps_per_epoch = 20000
seq_index       = 0
ses_index       = 0

data_attention = np.ones(data.shape)

def save_data_attention(data_attention, filename):
    print(f'Saving data_attention to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(data_attention, f)

zeros = np.zeros_like(data)
ones  = np.ones_like(data)
data_attention = np.where(np.all(data > 0, axis=-1, keepdims=True), ones, zeros)

test_data  = data[-1:]
train_data = data[:-1]

test_attention  = data_attention[-1:]
train_attention = data_attention[:-1]


def read_one(data):
    global seq_index, ses_index
    sample = torch.tensor(data[ses_index][seq_index:seq_index+sequence_size])
    seq_index += sequence_size
    if seq_index + sequence_size >= data.shape[1]:
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
def pytorch_train_on_data(model, optimizer, loss_fn, epochs, batch_size, steps_per_epoch):
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

            output = torch.reshape(output, (batch.shape[0], 1, NUM_PLAYERS, 24))
            target = torch.reshape(batch[:, -1], (batch.shape[0], 1, NUM_PLAYERS, 24))
            loss = loss_fn(output, target)
            if not step % 10:
                wandb.log({'loss': float(loss)})
                print(f'\tStep {step} loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        eval(model, optimizer, loss_fn, batch_size, steps_per_epoch)

def eval(model, optimizer, loss_fn, batch_size, steps):
    print(f'Evaluating model')
    with torch.no_grad():
        batch = read_batch(train_data)
        batch = batch.to(DEVICE)

        batch_flat = torch.reshape(batch, (batch.shape[0], batch.shape[1], batch.shape[2] * batch.shape[3]))
        output = model(batch_flat[:, :-1])
        output = torch.reshape(output, (batch.shape[0], 1, NUM_PLAYERS, 24))

        # loss = loss_fn(output[:, :-1], batch[:, 1:])
        target = torch.reshape(batch[:, -1], (batch.shape[0], 1, NUM_PLAYERS, 24))
        loss = loss_fn(output, target)

        wandb.log({'eval_loss': float(loss)})
        print(f'\tTraining loss: {loss.item()}')


def pytorch_define_lstm_model(input_size, output_size):
    model = torch.nn.Sequential(
        Bottleneck(latent_size=10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(20, 1*NUM_PLAYERS*24),
        torch.nn.ReLU(),
    )
    model.to(DEVICE)
    return model

def prepare_embedding():
    embedding = torch.nn.Embedding(num_embeddings=NUM_PLAYERS, embedding_dim=1)
    for player in range(NUM_PLAYERS):
        embedding.weight.data[player, 0] = torch.tensor([player])
    return embedding

wandb.init(name="VRCTrackingModel", project="VRCTrackingModel")

model       = pytorch_define_lstm_model(input_size=24, output_size=24)
optimizer   = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn     = torch.nn.MSELoss()
embedding   = prepare_embedding() #torch.nn.Embedding(num_embeddings=NUM_PLAYERS, embedding_dim=1)


print(f'Steps per epoch: {steps_per_epoch}')
print(f'Batch size: {batch_size}')
print(f'Training on {num_epochs} epochs')
print(f'Device: {DEVICE}')
print(f'Loss_fn: {loss_fn}')
pytorch_train_on_data(model=model, optimizer=optimizer,
                      loss_fn=loss_fn, epochs=num_epochs, batch_size=batch_size,
                      steps_per_epoch=steps_per_epoch)

output = { 'model': model,
           'offsets': offsets,
           'worldUUID': worldUUID,
           'sessionStart': sessionStart
          }

def save_model_pkl(model, model_filename):
    print(f'Saving model to {model_filename}')
    pickle.dump(model, open(model_filename, 'wb'))

save_model_pkl(output, MODEL_NAME + '.pkl')