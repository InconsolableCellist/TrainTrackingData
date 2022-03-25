from DataInjestor import DataInjestor
import os
import pickle

MAX_TIMESTEPS = 1250
MAX_PLAYERS   = 30
DATAFILE_NAMES = []
for file in os.listdir('input'):
    if file.endswith('.json'):
        DATAFILE_NAMES.append(file)
DATASET_NAME = "blackcatlocalposition"
DATAFILE_NAMES = sorted(DATAFILE_NAMES)

di = DataInjestor(max_timesteps = MAX_TIMESTEPS, max_players = MAX_PLAYERS, datafile_names = DATAFILE_NAMES)
batches = []
for i in range(0, len(DATAFILE_NAMES) - 1):
    di.process_datafile(DATAFILE_NAMES[i], i)

print(di.data.shape)

print("----\nThere are " + str(len(di.data)) + " sessions of data")
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
