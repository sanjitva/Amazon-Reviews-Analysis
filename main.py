import numpy as np
from flask import Flask,redirect,url_for,render_template,request
import joblib

#Instantiate WSGI Application
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        gender=request.form['gender']
        senior=request.form['senior']
        partner=request.form['partner']
        dependent=request.form['dependent']
        months=request.form['months']
        phone=request.form['phone']
        multiple_lines=request.form['multiple_lines']
        internet=request.form['internet']
        security=request.form['security']
        backup=request.form['backup']
        protection=request.form['protection']
        support=request.form['support']
        tv=request.form['tv']
        movies=request.form['movies']
        paperless=request.form['paperless']
        payment_method=request.form['payment_method']
        monthly_charge=request.form['monthly_charge']
        loyalty=request.form['loyalty']

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
        print(model_prediction)

        if model_prediction==0:
            return render_template('prediction.html',prediction_text="This customer is NOT Likely to Churn")
        else:
            return render_template('prediction.html',prediction_text="This customer is Likely to Churn")

            


if __name__ == '__main__':
    app.run(debug=True)