from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# Load the Random Forest CLassifier model
filename = 'result.pkl'
mayuri = pickle.load(open(filename, 'rb'))
@app.route('/',methods=['GET'])
def home():
	return render_template('lung.html')
standard_to = StandardScaler()


@app.route('/predict', methods=['POST'])
def predict():
    if  request.method == 'POST':
        AGE = int(request.form['AGE'])
        SMOKING = int(request.form['SMOKING'])
        YELLOW_FINGERS = int(request.form['YELLOW_FINGERS'])
        ANXIETY = int(request.form['ANXIETY'])
        PEER_PRESSURE= int(request.form['PEER_PRESSURE'])
        CHRONIC_DISEASE = int(request.form['CHRONIC_DISEASE'])
        FATIGUE	 = int(request.form['FATIGUE'])
        ALLERGY = int(request.form['ALLERGY'])
        WHEEZING = int(request.form['WHEEZING'])
        ALCOHOL_CONSUMING = int(request.form['ALCOHOL_CONSUMING'])
        COUGHING = int(request.form['COUGHING'])
        SHORTNESS_OF_BREATH = int(request.form['SHORTNESS_OF_BREATH'])
        SWALLOWING_DIFFICULTY = int(request.form['SWALLOWING_DIFFICULTY'])
        CHEST_PAIN = int(request.form['CHEST_PAIN'])
        GENDER= request.form['GENDER']
        if(GENDER=='M'):
            GENDER_M=1
        else:
            GENDER_M=0
        
        my_prediction = mayuri.predict([[AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN,GENDER_M]])
        output=(my_prediction[0])

        if int(output)<=0:
            return render_template('lung.html',my_prediction_texts="Good News You Don't Have Lung Cancer")
        else:
            return render_template('lung.html',my_prediction_texts="Sorry You Are Having Symptoms For Lung Cancer")
    else:
        return render_template('lung.html')

if __name__ == '__main__':
	app.run(debug=True)

