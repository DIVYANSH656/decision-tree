from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import numpy as np
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
app = Flask(__name__)
@app.route('/',methods=["GET"])
@cross_origin()
def homePage():
    return render_template("index.html")
@app.route('/predict',methods=["POST","GET"])
@cross_origin()
def index():
    if request.method == "POST":
        try:
            Pclass = request.form['Pclass']
            sex = request.form['sex']
            age = request.form['age']
            Sibsp = request.form['Sibsp']
            Parch = request.form['Parch']
            Fare = np.log1p(float(request.form['Fare']))
            filename = 'DecisionTreeAssignment.sav'
            loaded_model = load(open(filename,"rb"))
            prediction = loaded_model.predict([[Pclass,sex,age,Sibsp,Parch,Fare]])
            return render_template('result.html', prediction=int(prediction[0]))
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')
if(__name__=="__main__"):
    app.run(debug=True, host='127.0.0.1', port=5000)