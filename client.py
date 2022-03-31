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

