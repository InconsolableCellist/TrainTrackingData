import json

import numpy

from DataInjestor import DataInjestor
import os
import pickle

MAX_TIMESTEPS = 10000 # will get trimmed per player
MAX_PLAYERS   = 30 # will not get trimmed
DATAFILE_NAMES = []
for file in os.listdir('input'):
    if file.endswith('.json'):
        DATAFILE_NAMES.append(file)
DATASET_NAME = "blackcatlocalposition"
DATAFILE_NAMES = sorted(DATAFILE_NAMES)

print(f'Processing local/global offsets across all sessions')
di = DataInjestor(max_timesteps = MAX_TIMESTEPS, max_players = MAX_PLAYERS, datafile_names = DATAFILE_NAMES)
for i in range(0, len(DATAFILE_NAMES) - 1):
    with open(os.path.join('input', DATAFILE_NAMES[i]), 'r') as f:
        data_f = json.load(f)
        di.set_positional_offset_range(data_f)

print(f'Offsets calculated. local min/max and global min/max are: ({di.min_local_offset}/{di.max_local_offset}), ({di.min_global_offset}/{di.max_global_offset})')
for i in range(0, len(DATAFILE_NAMES) - 1):
    di.process_datafile(DATAFILE_NAMES[i], i)

print("----\nAll data loaded.\nThere are " + str(len(di.data)) + " sessions of data")
print(f'data shape for all {len(di.data)} sessions:')
for session in di.data:
    print(f'\t{session.shape}')

print(f'Offsets: local min/max and global min/max are: ({di.min_local_offset}/{di.max_local_offset}), ({di.min_global_offset}/{di.max_global_offset})')

print(f'Saving data to dataset/{DATASET_NAME}.pkl')
if not os.path.exists('dataset'):
    os.mkdir('dataset')

output_data = {'data': di.data,
               'offsets' : {
                   'max_global_offset': di.max_global_offset,
                   'max_local_offset': di.max_local_offset,
                   'min_global_offset': di.min_global_offset,
                   'min_local_offset': di.min_local_offset
               },
               'worldUUID': di.worldUUID,
               'sessionStart': di.sessionStart,
               }

with open(os.path.join('dataset', DATASET_NAME + '.pkl'), 'wb') as f:
    pickle.dump(output_data, f)
# np.save(os.path.join('dataset', DATASET_NAME + '.npy'), data)
# with open(os.path.join('dataset', DATASET_NAME + '.json'), 'w') as f:
#     json.dump(output_data['data'].tolist(), f)
