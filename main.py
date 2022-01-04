from flask import Flask, render_template, request
import pickle

diabebtes = pickle.load(open("diabetes_pipeline.pkl", "rb"))
heart_fail = pickle.load(open("heart_disease_pipeline.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes_prediction_input.html")

@app.route("/diabetes/diabetes_prediction", methods = ["POST","GET"])
def diabetes_prediction():
    input_val = [[float (X) for X in request.form.values()]]
    output = diabebtes.predict(input_val)
    prob = diabebtes.predict_proba(input_val)
    return render_template("diabetes_prediction_output.html", prediction = output, probabilty = prob)

@app.route("/heart")
def heart():
    return render_template("heart_failure_input.html")

@app.route("/heart/heart_failure", methods = ["POST","GET"])
def heart_failure():
    input_val = [[X for X in request.form.values()]]
    output = heart_fail.predict(input_val)
    prob = heart_fail.predict_proba(input_val)
    return render_template("heart_failure_output.html", prediction = output, probabilty = prob)

if __name__ == "__main__":
    app.run(debug=True)