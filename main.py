import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, url_for



app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
gender_le = pickle.load(open('gender_le.pkl', 'rb'))
married_le = pickle.load(open('married_le.pkl', 'rb'))
education_le = pickle.load(open('education_le.pkl', 'rb'))
self_employed_le = pickle.load(open('self_employed_le.pkl', 'rb'))
property_area_le = pickle.load(open('property_area_le.pkl', 'rb'))
loan_status_le = pickle.load(open('loan_status_le.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html', data="HELLO")


@app.route("/predict", methods=['POST'])
def predict():
    gender = request.form['gender']
    married = request.form['married']
    dependants = int(request.form['dependants'])
    education = request.form['education']
    self_employed = request.form['self_employed']
    applicant_income = int(request.form['applicant_income'])
    coapplicant_income = int(request.form['coapplicant_income'])
    loan_amount = int(request.form['loan_amount'])
    loan_amount_term = int(request.form['loan_amount_term'])
    credit_history = int(request.form['credit_history'])
    property_area = request.form['property_area']

    gender = gender_le.transform([gender])
    married = married_le.transform([married])
    education = education_le.transform([education])
    self_employed = self_employed_le.transform([self_employed])
    property_area = property_area_le.transform([property_area])

    data = np.array([[gender, married, dependants, education, 
    self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])

    my_prediction = int(model.predict(data))

    my_prediction = loan_status_le.inverse_transform([my_prediction])[0]

    return render_template('prediction.html', data=my_prediction)
  
if __name__ == '__main__':
    app.run(debug=True)

