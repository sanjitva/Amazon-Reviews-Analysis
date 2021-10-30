from flask import Flask, render_template, url_for,request
import numpy as np
from sklearn.externals import joblib
import pandas as pd


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/h1/calc')
def calc():
    return render_template("home.html")



@app.route('/home')
def stat():
    return render_template("h1.html")



@app.route("/predict", methods=['GET','POST'])

def predict():
    global pred_args
    if request.method == 'POST':
        try:
            District = ["kolhapur","Nashik","Satara","Sangli","Solapur"]
            District = request.form['District']
            if District == 'kolhapur':
                kolhapur = 1
                Nashik = 0
                Satara = 0
                Sangli = 0
                Solapur = 0
            if District == 'Nashik':
                kolhapur = 0
                Nashik = 1
                Satara = 0
                Sangli = 0
                Solapur = 0
            if District == 'Satara':
                kolhapur = 0
                Nashik = 0
                Satara = 1
                Sangli = 0
                Solapur = 0
            if District == 'Sangli':
                kolhapur = 0
                Nashik = 0
                Satara = 0
                Sangli = 1
                Solapur = 0
            if District == 'Solapur':
                kolhapur = 0
                Nashik = 0
                Satara = 0
                Sangli = 0
                Solapur = 1
            Year = float(request.form['Year'])
            Area = float(request.form['Area'])
            Production = float(request.form['Production'])
            Annual_rainfall = float(request.form['Annual_rainfall'])
            Temperature = float(request.form['Temperature'])
            pred_args = [kolhapur, Nashik, Satara, Sangli, Solapur, Year, Area, Production, Annual_rainfall, Temperature]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            mul_reg = open("multiple_regression_rice_model.pkl","rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
                      
        except ValueError:
                return "Please check the entered values"
        return render_template('predict.html', prediction = model_prediction )
               

	

if __name__  == '__main__' :
    app.run(debug=True)