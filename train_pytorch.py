import pickle, os
import numpy as np
import torch
import absl
from torch.utils.data import DataLoader

from VRCDataset import VRCDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
DEVICE      = "cuda:0"

DATASET_FILE = 'blackcatlocalposition.pkl'
DATASET_PATH = 'dataset'

dataset = {}
with open(os.path.join(DATASET_PATH, DATASET_FILE), 'rb') as f:
    data = pickle.load(f)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def ConvertndArraytoTensor(ndarray):
    return torch.from_numpy(ndarray).float().to(DEVICE)

# Gets one sampling of data for a player
def GetOneTimeslice(data, timeslice, player_num):
    return data[timeslice][player_num]

# Gets n seconds of data across all players
def BatchGetNSeconds(seconds, data):
    d = []
    for time in range(starting_timeslice, seconds * 3):
        for player_num in range(0, data.shape[1]):
            if time < data.shape[0]:
                d.append(GetOneTimeslice(data, time, player_num))
    return d, time


def ConvertListToTensor(data):
    return torch.tensor(data, dtype=torch.float32).to(DEVICE)

# def get_batch(last_timeslice, in_data, timeslice_size):
#     return [ BatchGetNSeconds(1, in_data, last_timeslice + e * timeslice_size) for e in range(batch_count) ], last_timeslice + batch_count * timeslice_size



def PytorchTrainOnData(in_data, model, optimizer, loss_fn, epochs, batch_size, device, steps_per_epoch):
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # data = torch.tensor(data, dtype=torch.float32).to(device)
            data = BatchGetNSeconds(1, in_data)
            optimizer.zero_grad()
            data = ConvertListToTensor(data)
            print(data.shape)
            prediction = model(data)
            # output = model(data)
            loss = loss_fn(data[:, 1:], prediction[:, :-1])
            loss.backward()
            optimizer.step()
            print("Epoch: " + str(epoch) + " Loss: " + str(loss))

def DefineTorchModel(input_size, hidden_size, output_size):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
        torch.nn.ReLU(),
    )
    model.to(DEVICE)
    return model


model = DefineTorchModel(input_size=24, hidden_size=10, output_size=24)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
num_epochs = 10
batch_size = 9 # 1 second = ~3 timeslices

# sanity checks
print("Number of sessions " + str(len(data)))
# for index in range(len(data)):
#     session = data[index]
#     print("Session " + str(index))
#     print("\tshape: " + str(np.shape(session)))
#     print("\tGetting one timeslice of data, for all players")
#     slice = GetOneTimeslice(session, 0, 0)
#     print("\tTimeslice has a shape of " + str(slice.shape))
#     print("\tGetting 10 seconds of data, for all players")
#     slice = np.array(BatchGetNSeconds(10, session, 0))
#     print("\t" + str(len(slice)) + " entries in the batch")
#     print("\tBatch has a shape of " + str(slice.shape))

# PytorchTrainOnData(data[0], model, optimizer, loss_fn, epochs, batch_size, DEVICE, 1000)


def GetNSeconds(dataiter, n):
    n *= 3
    data = []
    try:
        for i in range(0, n):
            data.append(dataiter.next())
    except StopIteration:
        pass
    print(f'data shape: {np.shape(data)}')
    return torch.tensor(data)


dataset = VRCDataset(json_file = DATASET_FILE, root_dir=DATASET_PATH)
for i, session in enumerate(dataset):
    print("on session " + str(i))
    dataloader = DataLoader(dataset=session, batch_size=batch_size, shuffle=True, num_workers=2)
    dataiter = iter(dataloader)
    print("\tshape of session: " + str(np.shape(session)))

    # batches = GetNSeconds(dataiter, batch_size)

    sequence_size = 1024
    batches = torch.tensor([GetNSeconds(dataiter, sequence_size) for e in range(batch_size)])

    print(f'\tGot {len(batches)} batches')
    # for batch in batches:
    #     total_samples = len(batch)
    #     print("\tshape of batch: " + str(np.shape(batch)))
