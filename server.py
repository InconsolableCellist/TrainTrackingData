import os
import pickle
import torch
from flask import Flask, request, jsonify
import numpy as np
from DataInjestor import DataInjestor

app = Flask(__name__)
context = None # shape: [session, time, players, data]
context_meta = None
model   = None
model_meta = None
MODEL_NAME = 'blackcatmodel-steps_100-batchsize_5000-epochs_75-latentsize_2048.pkl'
MODEL_PATH = 'models'
MAX_TIMESTEPS = 1250
NUM_PLAYERS = 30

ai_playerUUID = ""
ai_avatarID = ""
ai_playernum = 0 # The index into the player dimension that shows where the AI character is contained
ai_timeslice = 0 # The index into the time dimension for the AI player

offsets = {} # stores [ min_global_offset, max_global_offset, min_local_offset, max_local_offset]
# these are used to un-normalize/scale the normalized data back into worldspace and proper local space

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#data shape:
# [ batch_num, time, player_num, tracking_data ]
# TODO: or maybe it's [session_num, batch_num, time, player_num, tracking_data]

# execution workflow
#   Server starts and loads model
#   User sets the context (@POST /context)
#       Server sets ai_playernum to the first available player_num
#   User provides a playerUUID and avatarID for the AI character and the character's initial position (@POST /player, @POST /position)
#       The playerUUID should match the training data, of the form "usr_...", avatarID of the form "avtr_..."
#   User requests a prediction (@GET /prediction)
#       Server runs the model with the context and the ai_playernum inserted into the context
#       Server increments ai_timeslice and inserts that prediction into the context at [0, ai_timeslice, ai_playernum]
#       Server returns all 24 tracking data values of [0, ai_timeslice, ai_playernum]

# gets x, y, and z from the array and returns it
def get_xyz(data, offset):
    x = data[offset]
    y = data[offset + 1]
    z = data[offset + 2]
    return x, y, z

# scales the x, y, and z, un-normalizing it, and returns it
def get_xyz_scaled(data, offset, min_offset, max_offset):
    x, y, z = get_xyz(data, offset)
    return f'{x * min_offset:.6f}, {y * min_offset:.6f}, {z * min_offset:.6f}'

# goes through each player and finds the first one that has all 0s for data, indicating it's not being used
def set_ai_playernum():
    global ai_playernum
    for playernum in range(0, NUM_PLAYERS):
        if np.all(context[0, :, playernum] == 0):
            ai_playernum = playernum
            break

# This context should contain all the tracked data for every real character in the world over some period of time
@app.route('/context', methods=['POST'])
def set_context():
    global context, context_meta, ai_playernum, ai_timeslice
    context_meta = request.json
    di = DataInjestor(max_timesteps=MAX_TIMESTEPS, max_players=NUM_PLAYERS)
    di.process_data(context_meta)
    session = di.data[0]
    context = session[np.newaxis, :, :, :]
    print(f'context.shape: {context.shape}') # [1, 1250, 30, 24]
    # context = torch.tensor(context).to(DEVICE)
    # context = torch.reshape(context, (context.shape[0], context.shape[1], context.shape[2] * context.shape[3]))
    set_ai_playernum()

    return { 'success': True , 'input_shape': session.shape,
             'max_timesteps': di.max_timesteps, 'max_players': di.max_players,
             'internal_shape': context.shape,
             'ai_playernum': ai_playernum, 'ai_timeslice': ai_timeslice,
             'ai_playerUUID': ai_playerUUID, 'ai_avatarID': ai_avatarID }

def get_data_from_output(d):
    out_d = { 'ai_timeslice' : ai_timeslice, 'ai_playernum' : ai_playernum, 'ai_playerUUID' : ai_playerUUID, 'data': [] }
    player = {}
    datum = 0
    player['playerInstancePosition'] = get_xyz_scaled(d[0, 0, ai_playernum], datum, offsets['min_global_offset'], offsets['max_global_offset'])
    datum += 3
    player['playerInstanceRotation'] = get_xyz_scaled(d[0, 0, ai_playernum], datum, 0, 360)
    datum += 3
    player['headPosition'] = get_xyz_scaled(d[0, 0, ai_playernum], datum, offsets['min_local_offset'], offsets['max_local_offset'])
    datum += 3
    player['headRotation'] = get_xyz_scaled(d[0, 0, ai_playernum], datum, 0, 360)
    datum += 3
    player['leftHandPosition'] = get_xyz_scaled(d[0, 0, ai_playernum], datum, offsets['min_local_offset'], offsets['max_local_offset'])
    datum += 3
    player['leftHandRotation'] = get_xyz_scaled(d[0, 0, ai_playernum], datum, 0, 360)
    datum += 3
    player['rightHandPosition'] = get_xyz_scaled(d[0, 0, ai_playernum], datum, offsets['min_local_offset'], offsets['max_local_offset'])
    datum += 3
    player['rightHandRotation'] = get_xyz_scaled(d[0, 0, ai_playernum], datum, 0, 360)
    out_d['data'].append(player)
    return out_d

# Runs the model and gets the next timeslice
@app.route('/prediction', methods=['GET'])
def get_prediction():
    global context, ai_timeslice, ai_playernum
    if type(context) is np.ndarray:
        context = torch.tensor(context).to(DEVICE)

    context = torch.reshape(context, (context.shape[0], context.shape[1], context.shape[2] * context.shape[3]))
    output = model(context)
    context = torch.reshape(context, (context.shape[0], context.shape[1], NUM_PLAYERS, 24))
    output = torch.reshape(output, (1, 1, NUM_PLAYERS, 24))
    print(f'output[0, 0, :] = {output[0, 0, :]}')

    ai_timeslice += 1
    out_d = get_data_from_output(output)
    context[0, ai_timeslice, ai_playernum] = output[0, 0, ai_playernum]
    print(f'context[0, {ai_timeslice}, {ai_playernum}]: {context[0, ai_timeslice, ai_playernum]}')
    print(f'output[0, 0, {ai_playernum}]: {output[0, 0, ai_playernum]}')

    return out_d

# sets ai_playerUUID and ai_avatarID from { 'playerUUID' : 'usr_...', 'avatarID', :'avtr_...' }
@app.route('/player', methods=['POST'])
def set_playerUUID():
    global ai_playerUUID, ai_avatarID
    succ = True
    try:
        ai_playerUUID = request.json['playerUUID']
        ai_avatarID = request.json['avatarID']
    except KeyError:
        succ = False

    if succ:
        return { 'success': True, 'playerUUID': ai_playerUUID, 'avatarID': ai_avatarID, 'context'
        : 'missing (@POST /context)' if context is None else 'OK' }
    return { 'success': False }

# sets the initial position of the AI character
@app.route('/position', methods=['POST'])
def set_position():
    global context, ai_timeslice, ai_playernum
    if context is None:
        return { 'success': False, 'context' : 'missing (@POST /context)' }
    data = request.json
    context[0, 0, ai_playernum] = list(
        DataInjestor.get_xyz_normalized(data['playerInstancePosition'], offsets['min_global_offset'], offsets['max_global_offset'])) + \
        list(DataInjestor.get_xyz_normalized(data['playerInstanceRotation'], 0, 360)) + \
        list(DataInjestor.get_xyz_normalized(data['headPosition'], offsets['min_local_offset'], offsets['max_local_offset'])) + \
        list(DataInjestor.get_xyz_normalized(data['headRotation'], 0, 360)) + \
        list(DataInjestor.get_xyz_normalized(data['leftHandPosition'], offsets['min_local_offset'], offsets['max_local_offset'])) + \
        list(DataInjestor.get_xyz_normalized(data['leftHandRotation'], 0, 360)) + \
        list(DataInjestor.get_xyz_normalized(data['rightHandPosition'], offsets['min_local_offset'], offsets['max_local_offset'])) + \
        list(DataInjestor.get_xyz_normalized(data['rightHandRotation'], 0, 360))

    return { 'success': True, 'context_normalized': context[0, 0, ai_playernum].tolist() }

# returns the ai_playernum
@app.route('/playernum', methods=['GET'])
def get_playernum():
    return jsonify({ 'ai_playernum' : ai_playernum})

@app.route('/get_all_ai_data', methods=['GET'])
def get_all_ai_data():
    return jsonify({ 'ai_playerUUID' : ai_playerUUID, 'ai_avatarID' : ai_avatarID, 'ai_timeslice' : ai_timeslice,
                     'data' : context[0, :, ai_playernum].tolist(), 'context_shape': context.shape })

def load_model(model_path, model_name):
    global model_meta, model, offsets
    print(f'Loading model from {model_path}/{model_name}')
    with open(os.path.join(model_path, model_name), 'rb') as f:
        model_meta = pickle.load(f)
        model   = model_meta['model']
        offsets = model_meta['offsets']

if __name__ == "__main__":
    load_model(MODEL_PATH, MODEL_NAME)
    app.run(host='127.0.0.1', debug=True)


###
"""
model_meta:
 'model': model,
           'offsets': offsets,
           'worldUUID': worldUUID,
           'sessionStart': sessionStart
          }
          
model = pytorch lstm

    
    
context data:
{
  "sessionStart": 1647231653.0,
  "worldUUID": "wrld_1b482eca-bede-4de8-88a8-bbb6ca7e24cd",
  "data": {
    "usr_cff9574d-5e52-4366-89e4-e75c1e7fb5bd": {
      "displayName": "user",
      "username": "user",
      "playerUUID": "usr_cff9574d-5e52-4366-89e4-e75c1e7fb5bd",
      "avatarID": "avtr_c0a7dc83-1422-4b84-bce6-46093f4d7c47",
      "tracking_data": [
        {
          "timestamp": 1647231653341,
          "headPosition": "(-0.856579, 2.190100, 3.358989)",
          "headRotation": "(23.663230, 132.017200, 358.667600)",
          "leftHandPosition": "(-0.762248, 1.983636, 3.403341)",
          "leftHandRotation": "(45.180050, 114.839400, 39.637870)",
          "rightHandPosition": "(-0.870394, 1.992562, 3.286986)",
          "rightHandRotation": "(40.641400, 134.762400, 303.007200)",
          "leftFootPosition": "(-0.786080, 1.509293, 3.545429)",
          "leftFootRotation": "(23.509530, 204.220500, 126.272900)",
          "rightFootPosition": "(-0.909950, 1.499221, 3.382288)",
          "rightFootRotation": "(22.682580, 30.185580, 238.393600)",
          "leftKneePosition": "(-0.914610, 1.485900, 3.414665)",
          "leftKneeRotation": "(0.000000, 134.036100, 0.000000)",
          "rightKneePosition": "(-0.914610, 1.485900, 3.414665)",
          "rightKneeRotation": "(0.000000, 134.036100, 0.000000)",
          "leftElbowPosition": "(-0.914610, 1.485900, 3.414665)",
          "leftElbowRotation": "(0.000000, 134.036100, 0.000000)",
          "rightElbowPosition": "(-0.914610, 1.485900, 3.414665)",
          "rightElbowRotation": "(0.000000, 134.036100, 0.000000)",
          "hipPosition": "(-0.916085, 1.919870, 3.397258)",
          "hipRotation": "(359.806000, 129.350000, 2.393322)",
          "playerInstancePosition": "(3.903553, 0.994997, 7.229539)",
          "playerInstanceRotation": "(0.033118, 66.259820, 0.157565)"
        },
        """
###

