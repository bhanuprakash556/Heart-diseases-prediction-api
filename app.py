# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('heart1.pk', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    
    df = pd.DataFrame([int_features])
    
    df1=df.replace(["male","female"], [1,0])
    
    
    
    final_features = np.array(df1)
    
    final_features = final_features.reshape(1,5)
  #  final_features = pd.DataFrame([final_features])
    
    if model.predict(final_features) ==[1]:
       predict = "chances of the heart attack is 75%"
    else:
       predict = "may not be have heart attack"
    

    
    

    
    
  
    return render_template('index.html',prediction=predict)


if __name__ == "__main__":
    app.run(debug=True)
