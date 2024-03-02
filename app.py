from flask import Flask ,jsonify,request,render_template
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)

clf = joblib.load('final_model.joblib')
scaler = joblib.load('standard_scaler.joblib')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['Post'])
def predict():
    to_predict_list = request.form.to_dict()
    user_input = (to_predict_list['user_input']).split(" ") 
    user_input = [[float(inp) for inp in user_input]]
    user_input =  scaler.transform(user_input)
    pred = clf.predict(user_input)

    if pred[0] == 0:
        prediction = "Setosa"
    elif pred[0] ==1:
        prediction = "Versicolor"

    else:
        prediction = "Virginica"
    return render_template('predict.html',prediction = prediction )


if __name__ =="__main__":
    app.run(host='0.0.0.0',port=5000)
 