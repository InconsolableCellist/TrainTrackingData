import pickle, os
import numpy as np
import torch
import absl

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
def BatchGetNSeconds(seconds, data, starting_timeslice):
    d = []
    for time in range(starting_timeslice, seconds * 3):
        for player_num in range(0, data.shape[1]):
            if time < data.shape[0]:
                d.append(GetOneTimeslice(data, time, player_num))
    return d

def PytorchTrainOnData(data, model, optimizer, loss_fn, epochs, batch_size, device):
    for epoch in range(epochs):
        optimizer.zero_grad()
        data = torch.tensor(data, dtype=torch.float32).to(device)
        output = model(data)
        loss = loss_fn(output, data)
        loss.backward()
        optimizer.step()
        print("Epoch: " + str(epoch) + " Loss: " + str(loss))

def DefineTorchModel(input_size, hidden_size, output_size):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
    )
    model.to(DEVICE)
    return model


model = DefineTorchModel(input_size=24, hidden_size=10, output_size=24)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
epochs = 10
batch_size = 1

# sanity checks
print("Number of sessions " + str(len(data)))
for index in range(len(data)):
    session = data[index]
    print("Session " + str(index))
    print("\tshape: " + str(np.shape(session)))
    print("\tGetting one timeslice of data, for all players")
    slice = GetOneTimeslice(session, 0, 0)
    print("\tTimeslice has a shape of " + str(slice.shape))
    print("\tGetting 10 seconds of data, for all players")
    slice = np.array(BatchGetNSeconds(10, session, 0))
    # print("\t" + str(len(slice)) + " entries in the batch")
    print("\tBatch has a shape of " + str(slice.shape))

PytorchTrainOnData(data, model, optimizer, loss_fn, epochs, batch_size, DEVICE)