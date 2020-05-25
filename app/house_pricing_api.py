import sys
from sklearn.externals import joblib
from flask import Flask, jsonify, request

import pandas as pd
import numpy as np

import traceback
app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
    if housing_model:
        try:
            json_req = request.json
            print('input request',json_req)
            prediction_req = pd.get_dummies(pd.DataFrame(json_req))
            prediction_req = prediction_req.reindex(columns=housing_model_columns, fill_value=0)
            prediction = housing_model.predict(prediction_req)
            print('prediction via api call : ', prediction)
            return jsonify({'prediction': str(prediction)})
        except:
            print(traceback._cause_message)
            return jsonify({'trace': traceback.format_exc()})

    else:
        print('Failed to load any trained model')


if __name__ == '__main__':
    try :
        print('starting the house pricing application')
        port = int(sys.argv[1])

    except:
        port = 8088

    housing_model = joblib.load('housing_model.pkl')
    housing_model_columns = joblib.load('housing_model_columns.pkl')
    print('starting the application in port',port)
    app.run(port=port, debug=True,host='0.0.0.0')
    print('started application in port',port)
