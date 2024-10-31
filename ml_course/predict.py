import pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from flask import Flask
from flask import request, jsonify

customer = {
  "job": "management", 
  "duration": 400, 
  "poutcome": "success"
}
model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
  model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
  dv = pickle.load(f_in)

app = Flask('client')


@app.route('/predict', methods=['POST'])
def predict():
  customer = request.get_json()
  X = dv.transform([customer])
  y_pred = model.predict_proba(X)[:, 1][0]

  client = y_pred >= 0.5

  result = {
    'client' : bool(client),
    'y_pred' : float(y_pred)
  }
  
  return jsonify(result)


if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=9696)


