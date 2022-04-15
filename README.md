# TrainTrackingData

## Goal

To train a machine learning model to spit out believable human motion that will be useful to puppet VRChat avatars. Data is ingested via tracking data obtained from VRChat itself using VRCGrabTrackingData. 

## Steps

Once you've obtained data:

* To prepare the dataset open `injectData.py` and check the variables at top before running it. Right now it's configure to read all the json files in `input/` and will produce `dataset/blackcatlocalposition_no_normalization.pkl`

* To train the model, open `train_pytorch.py` and again check the top. It'll read `dataset/blackcatlocalposition_no_normalization.pkl` and eventually save it as `models/blackcatmodel_no_normalization-steps_100-batchsize_5000-epochs_50-latentsize_3072.pkl`

* Getting predictions is done with a Flask server right now. The server is `server.py` and you run it with flask via the CLI:

```
export FLASK_APP=server
flask run --host=127.0.0.1
```

* `client.py`  you can run normally. It hits the server with `/load_model`, `/context`, then a loop of `/prediction` and `/context` 512 times. It saves the returned data in `output/prediction.json`

* After all that, take `prediction.json` and put it into the Unity visualization program to see what it looks like more easily