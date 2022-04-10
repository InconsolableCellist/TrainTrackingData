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
        players[k] = { 'displayName': player['displayName'], 'userName': player['userName'],
                       'playerUUID': player['playerUUID'], 'avatarID': player['avatarID'],
                       'tracking_data': data }
        print(f'added {k} with {len(data)} entries')

    context = { 'data': players, 'worldUUID': context_meta['worldUUID'], 'sessionStart': context_meta['sessionStart'] }
    return context

def update_context(players, new_data):
    for k in players:
        player = players[k]
        player['tracking_data'] = player['tracking_data'][1:]
        if new_data['ai_playerUUID'] == player['playerUUID']:
            player['tracking_data'].append(new_data['data'][0])
            player['tracking_data'][-1]['timestamp'] = player['tracking_data'][-2]['timestamp'] + 300

            # player['tracking_data'].append(new_data['data'])
            # print(f'Updated ai player {k} with {new_data["data"]}')
        else:
            # print(f'updating normal player with the last value, because no real data yet')
            if len(player['tracking_data']) > 0:
                player['tracking_data'].append(player['tracking_data'][-1])
            else:
                player['tracking_data'].append({'timestamp': context_meta['sessionStart'] * 1000, 'playerInstancePosition' : "(0, 0, 0)",
                                                'playerInstanceRotation' : "(0, 0, 0)",
                                                'headPosition' : "(0, 0, 0)", 'headRotation' : "(0, 0, 0)",
                                                'leftHandPosition' : "(0, 0, 0)", 'leftHandRotation' : "(0, 0, 0)",
                                                'rightHandPosition' : "(0, 0, 0)", 'rightHandRotation' : "(0, 0, 0)",
                                                'leftFootPosition' : "(0, 0, 0)", 'leftFootRotation' : "(0, 0, 0)",
                                                'rightFootPosition' : "(0, 0, 0)", 'rightFootRotation' : "(0, 0, 0)",
                                                'leftKneePosition' : "(0, 0, 0)", 'leftKneeRotation' : "(0, 0, 0)",
                                                'rightKneePosition' : "(0, 0, 0)", 'rightKneeRotation' : "(0, 0, 0)",
                                                'leftElbowPosition' : "(0, 0, 0)", 'leftElbowRotation' : "(0, 0, 0)",
                                                'rightElbowPosition' : "(0, 0, 0)", 'rightElbowRotation' : "(0, 0, 0)",
                                                'hipPosition' : "(0, 0, 0)", 'hipRotation' : "(0, 0, 0)" })

context = get_initial_context()

def call_set_context():
    r = requests.post(PREDICTION_SERVER + '/context', json=context)
    print(f'Response: {r.status_code}')
    if r.status_code != 200:
        print(f'Error: {r.text}')
        exit(1)
    print(f'Sucessfully set the context')

call_set_context()

player_info = { 'playerUUID': 'usr_cff9574d-5e52-4366-89e4-e75c1e7fb5bd',
                'avatarID': 'avtr_c0a7dc83-1422-4b84-bce6-46093f4d7c47',
                'ai_playernum': 0 }
r = requests.post(PREDICTION_SERVER + '/player', json=player_info)
print(f'Response: {r.status_code}')
if r.status_code != 200:
    print(f'Error: {r.text}')
    exit(1)

print(f'Sucessfully set the player')

def call_get_prediction():
    print(f'Getting one prediction')
    r = requests.get(PREDICTION_SERVER + '/prediction')
    print(f'Response: {r.status_code}')
    if r.status_code != 200:
        print(f'Error: {r.text}')
        exit(1)
    print(f'Got prediction: {r.text}')

    return r.json()

prediciton = r.json()
for i in range(0, 512):
    new_data = call_get_prediction()
    update_context(context['data'], new_data)
    call_set_context()

def save_prediction_to_json(context):
    succ, error = DataInjestor.check_format(context)
    if not succ:
        print(f'Internal error in data format: {error}')

    f = open(os.path.join('output', 'prediction.json'), 'w')
    json.dump(context, f)
    f.close()
    print(f'Saved prediction to {os.path.join("output", "prediction.json")}')

save_prediction_to_json(context)