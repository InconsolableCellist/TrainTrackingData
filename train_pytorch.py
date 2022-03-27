import pickle, os
from random import random

import numpy as np
import torch
import time
import wandb

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

data    = np.asarray(input['data'])
offsets = input['offsets']
worldUUID    = input['worldUUID']
sessionStart = input['sessionStart']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sequence_size   = 32 # ~3 per second
num_epochs      = 75
batch_size      = 5000
steps_per_epoch = 100
latent_size     = 2048
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
            # if not step % 10:
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


def pytorch_define_lstm_model(input_size, output_size, latent_size):
    model = torch.nn.Sequential(
        Bottleneck(latent_size=latent_size),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size, latent_size),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size, latent_size//2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size//2, latent_size//4),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(latent_size//4, 1*NUM_PLAYERS*24),
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

model       = pytorch_define_lstm_model(input_size=24, output_size=24, latent_size=latent_size)
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
    if os.path.exists(model_filename + '.pkl'):
        model_filename = model_filename + '-' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
    else:
        model_filename = model_filename + '.pkl'
    print(f'Saving model to {model_filename}')
    pickle.dump(model, open(model_filename, 'wb'))

filename = f'{MODEL_NAME}-steps_{steps_per_epoch}-batchsize_{batch_size}-epochs_{num_epochs}-latentsize_{latent_size}'
save_model_pkl(output, filename)