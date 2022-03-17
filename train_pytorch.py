import pickle, os
from random import random

import numpy as np
import torch
import absl
import wandb
from torch.utils.data import DataLoader

from VRCDataset import VRCDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB_MODE"] = "offline"
DEVICE      = "cuda:0"

DATASET_FILE = 'blackcatlocalposition.pkl'
DATASET_PATH = 'dataset'

dataset = {}
with open(os.path.join(DATASET_PATH, DATASET_FILE), 'rb') as f:
    data = pickle.load(f)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size      = 16
sequence_size   = 2048
seq_index       = 0
ses_index       = 0

data_attention = np.ones(data.shape)

def save_data_attention(data_attention, filename):
    print(f'Saving data_attention to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(data_attention, f)

if not os.path.exists('data_attention.pkl'):
    print("exhaustively setting attention for padding because I can't think of another way to do it")
    for session in range(0, data.shape[0]):
        for time in range(0, data[session].shape[0]):
            for player in range(0, data[session][time].shape[0]):
                if np.all((data[session][time][player] == 0)):
                    data_attention[session][time][player] = np.zeros(24)
    save_data_attention(data_attention, 'data_attention.pkl')
else:
    with open('data_attention.pkl', 'rb') as f:
        data_attention = pickle.load(f)
    print("loaded data_attention.pkl. make sure it's not out of date")

zeros = np.zeros_like(data)
ones  = np.ones_like(data)
data_attention = np.where(np.all(data > 0, axis=-1, keepdims=True), ones, zeros)

test_data  = data[-1:]
train_data = data[:-1]

test_attention = data_attention[-1:]
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
print(f'read_one shape: {read_one(train_data).shape}')
print(f'read_batch shape: {read_batch(train_data).shape}')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def pytorch_train_on_data(model, optimizer, loss_fn, epochs, batch_size, steps_per_epoch):
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for step in range(steps_per_epoch):
            batch = read_batch(train_data)
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output[:, :-1], batch[:, 1:])
            wandb.log({'loss': float(loss)})
            loss.backward()
            optimizer.step()
            print(f'\tStep {step} loss: {loss.item()}')
        eval(model, optimizer, loss_fn, batch_size, steps_per_epoch)

def eval(model, optimizer, loss_fn, batch_size, steps):
    print(f'Evaluating model')
    for step in range(steps):
        batch = read_batch(train_data)
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output[:, :-1], batch[:, 1:])
        wandb.log({'training_loss': float(loss)})
        optimizer.step()
        print(f'\tTraining step {step} loss: {loss.item()}')


def pytorch_define_model(input_size, output_size):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, output_size),
        torch.nn.ReLU(),
    )
    model.to(DEVICE)
    return model

wandb.init(name="VRCTrackingModel", project="VRCTrackingModel")

model = pytorch_define_model(input_size=24, output_size=24)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
num_epochs = 3
batch_size = 9
steps_per_epoch = 100
print(f'Steps per epoch: {steps_per_epoch}')
print(f'Batch size: {batch_size}')
print(f'Training on {num_epochs} epochs')
print(f'Device: {DEVICE}')
print(f'Loss_fn: {loss_fn}')
pytorch_train_on_data(model=model, optimizer=optimizer,
                      loss_fn=loss_fn, epochs=num_epochs, batch_size=batch_size,
                      steps_per_epoch=steps_per_epoch)