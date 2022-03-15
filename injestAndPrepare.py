
###
"""
{
  "sessionStart": 1647231653.0,
  "worldUUID": "wrld_1b482eca-bede-4de8-88a8-bbb6ca7e24cd",
  "data": {
    "usr_cff9574d-5e52-4366-89e4-e75c1e7fb5bd": {
      "displayName": "user",
      "userName": "user",
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
          "hipRotation": "(359.806000, 129.350000, 2.393322)"
        },
        """
###

import tensorflow as tf
import numpy as np
import json, pickle, os


# (batch, time interval, player, playerdata)
# playerdata = headpos x, y, z, headrot x y z, lefthand x, y, z, lefthandrot x, y, z, righthandpos x, y, z, righthandrot x, y z
MAX_TIMESTEPS = 10000
MAX_PLAYERS = 100
data = np.zeros((MAX_TIMESTEPS, MAX_PLAYERS, 18), dtype=np.float32)

def get_xyz(tupleString):
    x, y, z = tupleString[1:-1].split(',')
    return float(x), float(y), float(z)

def get_xyz_normalized(tupleString, min_offset, max_offset):
    x, y, z = get_xyz(tupleString)
    return (x - min_offset) / (max_offset - min_offset), (y - min_offset) / (max_offset - min_offset), \
           (z - min_offset) / (max_offset - min_offset)

def get_positional_offset_range(d):
    max_offset = -1
    min_offset = int(1e9)
    for entry in d['data']:
        for td in d['data'][entry]['tracking_data']:
            for val in ['headPosition', 'leftHandPosition', 'rightHandPosition']:
                a = max(get_xyz(td[val]))
                b = min(get_xyz(td[val]))
                if a > max_offset:
                    max_offset = a
                if b < min_offset:
                    min_offset = b
    return (min_offset, max_offset)

avg_time_offset = np.zeros(MAX_PLAYERS)
last_time = 0
with open('sovreignschillhome.json') as json_file:
    data_f = json.load(json_file)
    time_start_ms = data_f['sessionStart'] * 1000
    player_num = 0
    vals_added = 0
    min_position, max_position = get_positional_offset_range(data_f)
    for entry in data_f['data']:
        if player_num > MAX_PLAYERS:
            break
        player = data_f['data'][entry]
        time = 0
        for td in player['tracking_data']:
            if time > MAX_TIMESTEPS:
                break
            data_time_ms = td['timestamp'] - time_start_ms
            if time > 0:
                avg_time_offset[player_num] += (data_time_ms - last_time)
            last_time = data_time_ms

            data[time][player_num] = list(get_xyz_normalized(td['headPosition'], min_position, max_position)) \
                                     + list(get_xyz_normalized(td['headRotation'], 0, 360)) \
                                     + list(get_xyz_normalized(td['leftHandPosition'], min_position, max_position)) \
                                     + list(get_xyz_normalized(td['leftHandRotation'], 0, 360)) \
                                     + list(get_xyz_normalized(td['rightHandPosition'], min_position, max_position)) \
                                     + list(get_xyz_normalized(td['rightHandRotation'], 0, 360))
            vals_added += 18

            time += 1
        avg_time_offset[player_num] /= abs(time - 1)
        player_num += 1

avg_time_offset = avg_time_offset[avg_time_offset != 0]

print("processed " + str(player_num) + " players")
print("added " + str(vals_added) + " values over " + str(time) + " timesteps covering " +
      '{:.2f} minutes'.format(np.average(avg_time_offset) / 1000 * time / 60))
print("Average time differential between datapoints is: " + '{:.4f} ms '.format(np.average(avg_time_offset)) +
      'with a standard deviation of ' + '{:.4f} ms '.format(np.std(avg_time_offset)))
print("min and max world positional data were: (" + str(min_position) + ", " + str(max_position) +
      "), normalized to 0.0 to 1.0")
print("min and max rotational (Euler) data were assumed to be 0 to 360, normalized to 0.0 to 1.0")

with open('sovreignschillhome.pkl', 'wb') as f:
    pickle.dump(data, f)

np.save('sovreignschillhome.npy', data)
