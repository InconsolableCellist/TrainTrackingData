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

f = open(os.path.join('input', 'TrackedData_wrld_4cf554b4-430c-4f8f-b53e-1f294eed230b_1648352371.json'), 'r')
context_meta = json.load(f)
f.close()

def get_initial_context():
    players = {}
    for k in context_meta['data']:
        player = context_meta['data'][k]
        i = 0
        data = []
        for td in player['tracking_data']:
            if i >= MAX_TIMESTEPS:
                break
            data.append(td)
            i += 1
        players[k] = { 'displayName': player['displayName'], 'username': player['userName'],
                       'playerUUID': player['playerUUID'], 'avatarID': player['avatarID'],
                       'tracking_data': data }
        print(f'added {k} with {len(data)} entries')

    context = { 'data': players, 'worldUUID': context_meta['worldUUID'], 'sessionStart': context_meta['sessionStart'] }
    return context

context = get_initial_context()

r = requests.post(PREDICTION_SERVER + '/context', json=context)
print(f'Response: {r.status_code}')
if r.status_code != 200:
    print(f'Error: {r.text}')
    exit(1)

print(f'Sucessfully set the context')

player_info = { 'playerUUID': 'usr_cff9574d-5e52-4366-89e4-e75c1e7fb5bd',
                'avatarID': 'avtr_c0a7dc83-1422-4b84-bce6-46093f4d7c47',
                'ai_playernum': 0 }
r = requests.post(PREDICTION_SERVER + '/player', json=player_info)
print(f'Response: {r.status_code}')
if r.status_code != 200:
    print(f'Error: {r.text}')
    exit(1)

print(f'Sucessfully set the player')

def get_prediction():
    print(f'Getting one prediction')
    r = requests.get(PREDICTION_SERVER + '/prediction')
    print(f'Response: {r.status_code}')
    if r.status_code != 200:
        print(f'Error: {r.text}')
        exit(1)
    print(f'Got prediction: {r.text}')

prediciton = r.json()
for i in range(0, 1024):
    get_prediction()
