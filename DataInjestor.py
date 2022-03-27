
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
          "hipRotation": "(359.806000, 129.350000, 2.393322)",
          "playerInstancePosition": "(3.903553, 0.994997, 7.229539)",
          "playerInstanceRotation": "(0.033118, 66.259820, 0.157565)"
        },
        """
###

import numpy as np
import json, pickle, os

# (batch, sequence/time, player, playerdata)

class DataInjestor:
    def __init__(self, max_timesteps, max_players, datafile_names=None):
        self.max_timesteps = max_timesteps
        self.max_players = max_players
        self.datafile_names = datafile_names
        self.data         = np.zeros((1 if datafile_names is None else len(datafile_names), max_timesteps, max_players, 24), dtype=np.float32)
        self.worldUUID    = ""
        self.sessionStart = 0.0
        self.players      = {}
        self.min_global_pos = int(1e9)
        self.max_global_pos = -1
        self.min_local_pos = int(1e9)
        self.max_local_pos = -1
        self.max_global_offset = -1
        self.max_local_offset = -1
        self.min_global_offset = int(1e9)
        self.min_local_offset = int(1e9)
        pass

    @staticmethod
    def get_xyz(tupleString):
        x, y, z = tupleString[1:-1].split(',')
        return float(x), float(y), float(z)

    @staticmethod
    def get_xyz_normalized(tupleString, min_offset, max_offset):
        x, y, z = DataInjestor.get_xyz(tupleString)
        return (x - min_offset) / (max_offset - min_offset), (y - min_offset) / (max_offset - min_offset), \
               (z - min_offset) / (max_offset - min_offset)


    # Iterates through all the data and finds the bounds (min and max) for the global position and local offsets
    # Then returns them as (min_global, max_global, min_local, max_local)
    def get_positional_offset_range(self, d):
        max_global_offset, max_local_offset, min_global_offset, min_local_offset = -1, -1, int(1e9), int(1e9)
        for entry in d['data']:
            for td in d['data'][entry]['tracking_data']:
                a = max(self.get_xyz(td['playerInstancePosition']))
                b = min(self.get_xyz(td['playerInstancePosition']))
                if a > max_global_offset:
                    max_global_offset = a
                if b < min_global_offset:
                    min_global_offset = b

                for val in ['headPosition', 'leftHandPosition', 'rightHandPosition']:
                    a = max(self.get_xyz(td[val]))
                    b = min(self.get_xyz(td[val]))
                    if a > max_local_offset:
                        max_local_offset = a
                    if b < min_local_offset:
                        min_local_offset = b
        self.max_global_offset = max_global_offset
        self.max_local_offset = max_local_offset
        self.min_global_offset = min_global_offset
        self.min_local_offset = min_local_offset
        return (min_global_offset, max_global_offset, min_local_offset, max_local_offset)

    def process_datafile(self, datafile, datafile_num):
        print(f'Processing {datafile}')
        with open(os.path.join('input', datafile)) as json_file:
            data_f = json.load(json_file)
            return self.process_data(data_f)

    def process_data(self, data_f, datafile_num=0):
        avg_time_offset = np.zeros(self.max_players)
        last_time = 0
        observed_timesteps = np.zeros(self.max_timesteps)
        time_start_ms = data_f['sessionStart'] * 1000
        player_num = 0
        vals_added = 0
        worldUUID = data_f['worldUUID']
        sessionStart = data_f['sessionStart']
        self.min_global_pos, self.max_global_pos, self.min_local_pos, self.max_local_pos = self.get_positional_offset_range(data_f)
        for entry in data_f['data']:
            if player_num > self.max_players:
                break
            player = data_f['data'][entry]
            self.players['playerUUID'] = player_num
            print(f'Processing player {player_num} with UUID {player["playerUUID"]}')
            time = 0
            for td in player['tracking_data']:
                if time > self.max_timesteps:
                    break
                data_time_ms = td['timestamp'] - time_start_ms
                if time > 0:
                    avg_time_offset[player_num] += (data_time_ms - last_time)
                last_time = data_time_ms

                self.data[datafile_num][time][player_num] = list(
                    self.get_xyz_normalized(td['playerInstancePosition'], self.min_global_pos, self.max_global_pos)) \
                                                            + list(self.get_xyz_normalized(td['playerInstanceRotation'], 0, 360)) \
                                                            + list(self.get_xyz_normalized(td['headPosition'], self.min_global_pos, self.max_global_pos)) \
                                                            + list(self.get_xyz_normalized(td['headRotation'], 0, 360)) \
                                                            + list(
                    self.get_xyz_normalized(td['leftHandPosition'], self.min_global_pos, self.max_global_pos)) \
                                                            + list(self.get_xyz_normalized(td['leftHandRotation'], 0, 360)) \
                                                            + list(
                    self.get_xyz_normalized(td['rightHandPosition'], self.min_global_pos, self.max_global_pos)) \
                                                            + list(self.get_xyz_normalized(td['rightHandRotation'], 0, 360))
                vals_added += 18

                time += 1
            observed_timesteps[player_num] = time
            avg_time_offset[player_num] /= abs(time - 1)
            player_num += 1
        avg_time_offset = avg_time_offset[avg_time_offset != 0]
        observed_timesteps = observed_timesteps[observed_timesteps != 0]

        print("processed " + str(player_num) + " players")
        print("added " + str(vals_added) + " values.")
        print(f'There was an average of {np.average(observed_timesteps):.1f} timesteps per player ' +
              f' with a standard deviation of {np.std(observed_timesteps):.2f} timesteps')
        print('Average time differential between datapoints is: {:.4f} ms '.format(np.average(avg_time_offset)) +
              'with a standard deviation of ' + '{:.4f} ms '.format(np.std(avg_time_offset)))
        print("min and max world positional data were: (" + str(self.min_global_pos) + ", " + str(self.max_global_pos) +
              "), normalized to 0.0 to 1.0")
        print("min and max local positional data were: (" + str(self.min_local_pos) + ", " + str(self.max_local_pos) +
              "), normalized to 0.0 to 1.0")
        print("min and max rotational (Euler) data were assumed to be 0 to 360, normalized to 0.0 to 1.0")
        print("numpy data shape: " + str(np.shape(self.data)))
        # print("stripping zeros.\nnumpy data shape: " + str(np.shape(data)))
        print(observed_timesteps)
        return self.data
