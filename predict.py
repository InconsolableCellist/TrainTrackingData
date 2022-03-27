import json

import torch, pickle, os
import numpy as np

MODEL_NAME  = 'blackcatmodel-steps_100-batchsize_5000-epochs_75-latentsize_2048.pkl'
MODEL_PATH  = 'models'
DATAFILE_NAME = 'blackcatlocalposition.pkl'
DATA_PATH   = 'dataset'
DEVICE      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_PLAYERS = 30

def open_model(filename):
    with open(filename, 'rb') as f:
        model = torch.load(f)
        print(f'Model loaded from {filename}')
    return model

def open_model_pkl(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        print(f'Model loaded from {filename}')
    return model

def open_data_file(path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
        data = pickle.load(f)
        print(f'Data loaded from {filename}')
    return data

def get_xyz(data, offset):
    x = data[offset]
    y = data[offset + 1]
    z = data[offset + 2]
    return x, y, z

def get_xyz_scaled(data, offset, min_offset, max_offset):
    x, y, z = get_xyz(data, offset)
    return f'{x * min_offset:.6f}, {y * min_offset:.6f}, {z * min_offset:.6f}'


model_meta = open_model_pkl(os.path.join(MODEL_PATH, MODEL_NAME))
model = model_meta['model']
input = open_data_file(DATA_PATH, DATAFILE_NAME)
data  = input['data']
offsets = input['offsets']

session = data[0]
print(f'session shape: {session.shape}')
context = session[np.newaxis, :10,:, :] # strip the data down to the first 10 samples and add a batch dim
context = torch.tensor(context).to(DEVICE)
print(f'batching context into one batch: {context.shape}')
context = torch.reshape(context, (context.shape[0], context.shape[1], context.shape[2] * context.shape[3]))
print(f'context reshaped to be: {context.shape}')
output = model(context)
print(f'output from the model: {output.shape}')
output = torch.reshape(output, (1, 1, NUM_PLAYERS, 24))
print(f'converting output to: {output.shape}')


def save_output(output, offsets, filename):
    global data
    if not os.path.exists('output'):
        os.makedirs('output')
    f = open(os.path.join('output', 'prediction.json'), 'w')
    out_d = { 'sesionStart' : model_meta['sessionStart'],
              'worldUUID' : model_meta['worldUUID'],
              'data' : [] }
    for playernum in range(0, output.shape[2]-1):
        player = {}
        datum = 0
        # player['playerInstancePosition'] = make_xyz(output[0][0][playernum], datum)
        player['playerInstancePosition'] = get_xyz_scaled(output[0][0][playernum], datum,
                                                          offsets['min_global_offset'],
                                                          offsets['max_global_offset'])
        datum += 3
        player['playerInstanceRotation'] = get_xyz_scaled(output[0][0][playernum], datum, 360, 360)
        datum += 3
        player['headPosition'] = get_xyz_scaled(output[0][0][playernum], datum,
                                                offsets['min_local_offset'],
                                                offsets['max_local_offset'])
        datum += 3
        player['headRotation'] = get_xyz_scaled(output[0][0][playernum], datum, 360, 360)
        datum += 3
        player['leftHandPosition'] = get_xyz_scaled(output[0][0][playernum], datum,
                                                    offsets['min_local_offset'],
                                                    offsets['max_local_offset'])
        datum += 3
        player['leftHandRotation'] = get_xyz_scaled(output[0][0][playernum], datum, 360, 360)
        datum += 3
        player['rightHandPosition'] = get_xyz_scaled(output[0][0][playernum], datum,
                                                     offsets['min_local_offset'],
                                                     offsets['max_local_offset'])
        datum += 3
        player['rightHandRotation'] = get_xyz_scaled(output[0][0][playernum], datum, 360, 360)
        out_d['data'].append(player)
    json.dump(out_d, f)
    f.close()

save_output(output, offsets, 'prediction.json')
print(f'output[0, 0]: {output[0, 0]}')



# print("10 timeslices for player 0")
# for i in range(0, 10):
#     print(f'timeslice: {i}')
#     x, y, z = get_xyz(session[i][0], 0)
#     print(f'\tplayerInstancePosition\n\t\tx: {x:.6f}, y: {y:.6f}, z: {z:.6f}')
#     x, y, z = get_xyz(session[i][0], 3)
#     print(f'\tplayerInstanceRotation\n\t\tx: {x:.6f}, y: {y:.6f}, z: {z:.6f}')
#     x, y, z = get_xyz(session[i][0], 6)
#     print(f'\theadPosition\n\t\tx: {x:.6f}, y: {y:.6f}, z: {z:.6f}')
#     x, y, z = get_xyz(session[i][0], 9)
#     print(f'\theadRotation\n\t\tx: {x:.6f}, y: {y:.6f}, z: {z:.6f}')
#     x, y, z = get_xyz(session[i][0], 12)
#     print(f'\tleftHandPosition\n\t\tx: {x:.6f}, y: {y:.6f}, z: {z:.6f}')
#     x, y, z = get_xyz(session[i][0], 15)
#     print(f'\tleftHandRotation\n\t\tx: {x:.6f}, y: {y:.6f}, z: {z:.6f}')
#     x, y, z = get_xyz(session[i][0], 18)
#     print(f'\trightHandPosition\n\t\tx: {x:.6f}, y: {y:.6f}, z: {z:.6f}')
#     x, y, z = get_xyz(session[i][0], 21)
#     print(f'\trightHandRotation\n\t\tx: {x:.6f}, y: {y:.6f}, z: {z:.6f}')
