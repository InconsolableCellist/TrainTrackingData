import json
from RingBuffer import RingBuffer
import requests

from DataInjestor import DataInjestor
import os

DATAFILE_NAME = "TrackedData_wrld_4cf554b4-430c-4f8f-b53e-1f294eed230b_1648352371.json"
DATAFILE_PATH = "input"
MAX_TIMESTEPS = 1024
MAX_PLAYERS   = 30
PREDICTION_SERVER = "http://localhost:5000"

buffer    = RingBuffer(MAX_TIMESTEPS)

def fetch_data(path, filename):
    f = open(os.path.join(path, filename), 'r')
    context_meta = json.load(f)
    di = DataInjestor(max_timesteps=MAX_TIMESTEPS, max_players=MAX_PLAYERS)
    context = di.process_data(context_meta)
    f.close()
    return context

def fill_buffer(context):
    for i in range(0, context[0].shape[0]):
        buffer.append(context[0][i])

# @POST to /context
def send_context(context):
    print("Sending context to server...")
    r = requests.post(PREDICTION_SERVER + "/context", json=context)
    print("Context sent to server.")
    if r.status_code != 200:
        print("Error: " + r.text)
        return None
    return r

# def convert_context_to_json():
    # j = json.dumps(buffer.get_all())
    # data = buffer.get()
    # for i in range(len(data)):
    #     timeslice = data[i]
    #     print(f'timeslice.shape: {timeslice.shape}')
    #     players = {}
    #     for player in timeslice:
    #         print(player.shape)
            # players[player] = value
            # print(f'{player} : {value}')
            # break


    # return j
    # j = None
    # return j

context = fetch_data(DATAFILE_PATH, DATAFILE_NAME)
print(f'context.len: {len(context)}')
print(f'context[0].shape: {context[0].shape}')
fill_buffer(context)

# j = convert_context_to_json()
# send_context(context[0])
