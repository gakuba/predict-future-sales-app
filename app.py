# app-1.py
from flask import Flask, request, jsonify
import os
# import subprocess
# import sys
import pickle
import pandas as pd
import time
import json
import numpy as np
# from fluent import sender
# from fluent import event

app = Flask(__name__)

# Set environment variables
os.environ['PREDICTION_URI'] = '/Users/GAKUBA/Documents/predictions'
os.environ['MODEL_URI'] = 'gs://facileai-dev-kubeflowpipelines-default/staging/model'
os.environ['MODEL_NAME'] = 'model.pkl'
os.environ['PROC_FILENAME'] = 'data-proc-obj.pkl'

PREDICTION_URI = os.getenv('PREDICTION_URI')
MODEL_URI = os.getenv('MODEL_URI')
MODEL_NAME = os.getenv('MODEL_NAME')
PROC_FILENAME = os.getenv('PROC_FILENAME')

print('Load the model...')

# subprocess.check_call(['gsutil', 'cp', '{}/{}'.format(MODEL_URI, MODEL_NAME), MODEL_NAME],
#                         stderr=sys.stdout)
  
with open(MODEL_NAME, 'rb') as model_file:
    model = pickle.load(model_file)

print('Load the data processor...')
# Copy the preproc file from GCS
# subprocess.check_call(['gsutil', 'cp', '{}/{}'.format(MODEL_URI, PROC_FILENAME), PROC_FILENAME],
#                     stderr=sys.stdout)

with open(PROC_FILENAME, 'rb') as preproc_file:
    preproc = pickle.load(preproc_file)




@app.route('/bulk', methods=['POST'])
def bulk():
    uri = request.get_json().get('data_uri')

    print('open the data....')
    X_new = pd.read_csv(uri,low_memory=False)
    # X_new = np.fromiter(args.values(), dtype=float)  # convert input to array

    print('process the data...')
    new_preproc = preproc.train.new(X_new)
    new_preproc.process()
    X_new_proc = new_preproc.train.xs

    print('predict from the data...')
    y_hat = model.predict(X_new_proc)

    X_new['predicted'] = y_hat
    
    # print('send the result to elasticsearch')
    # sender.setup('python', host='localhost', port=24224)
    # # event.Event('predictions', {'data':X_new.to_json(orient="records")})
    # event.Event('predictions',{"time": "2020-06-22 15:13:21,300", "level": "INFO", "message": "Sending Email to username: 'Jack Sparrow' regarding server_ip: '192.168.1.2' ","myname":"Elie"})
    prediction_file_name = '{}/{}.csv'.format(PREDICTION_URI,int(time.time()))
    X_new.to_csv(prediction_file_name,index=False)

    print('return the response to the client...')
    out = {'location': prediction_file_name}
    
    return out, 200

    
@app.route('/predict', methods=['POST'])
def predict():
    
    X_new = request.get_json().get('observations')
    
    print('open the data....')
    X_new = json.dumps(X_new, indent = 4)
    
    X_new = pd.read_json(X_new, orient='split')

    print('process the data...')
    print(X_new.shape)
    new_preproc = preproc.train.new(X_new)
    new_preproc.process()
    X_new_proc = new_preproc.train.xs

    print('predict from the data...')
    y_hat = model.predict(X_new_proc).tolist()
    print('return the response to the client...')
    out = {'timestamp': int(time.time()), 'prediction':y_hat}
    return out, 200    

if __name__ == '__main__':
    app.run(debug=True)