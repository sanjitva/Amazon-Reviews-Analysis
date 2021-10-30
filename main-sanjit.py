import numpy as np
from flask import Flask,redirect,url_for,render_template
import joblib

#Instantiate WSGI Application
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    result=0
    if request.method=='POST':
        gender=requests.form(['gender'])
        senior=requests.form(['senior'])
        partner=requests.form(['partner'])
        dependent=requests.form(['dependent'])
        months=requests.form(['months'])
        phone=requests.form(['phone'])
        multiple_lines=requests.form(['multiple_lines'])
        internet=requests.form(['internet'])
        security=requests.form(['security'])
        backup=requests.form(['backup'])
        protection=requests.form(['protection'])
        support=requests.form(['support'])
        tv=requests.form(['tv'])
        movies=requests.form(['movies'])
        paperless=requests.form(['paperless'])
        payment_method=requests.form(['payment_method'])
        monthly_charge=requests.form(['monthly_charge'])
        loyalty=requests.form(['loyalty'])

        #Taking all the inputs above and putting them in a list
        pred_args = [gender,senior,partner,dependent,months,phone,multiple_lines,
                    internet,security,backup,protection,support,tv,movies,
                    paperless,payment_method,monthly_charge,loyalty]

        #Converting the list into an array
        pred_args_arr = np.array(pred_args)

        pred_args_arr = pred_args_arr.reshape(1,-1)

        #Opening the pickled model
        knn_model = open("knn_model.pkl","rb")

        #Loading joblib to call the pickled model
        ml_model = joblib.load(knn_model)

        #Deriving Predictions from the pickled model
        model_prediction = ml_model.predict(pred_args_arr)

        return render_template('predict.html', prediction = model_prediction )

            


if __name__ == '__main__':
    app.run(debug=True)