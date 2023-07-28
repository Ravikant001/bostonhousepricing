import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

#create flask app
app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regression.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

# Defines the route of the app
@app.route('/')
def home():
    return render_template('home.html')

# For sending request to app for doing the task
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        data_value = data['data']
        print(data_value)
        print(np.array(list(data_value.values())).reshape(1, -1))
        new_data = scalar.transform(np.array(list(data_value.values())).reshape(1, -1))
        output = regmodel.predict(new_data)
        print(output[0])
        return jsonify({'prediction': output[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
