#import libraries
#numpy is for numerical applications
#flask is for web applications
#pickle is the file we want it to read
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
#rb to read the model
flask_app = Flask(__name__)
model = pickle.load(open("crop_model.pkl", "rb"))

#user visit this route so it renders index.html
@flask_app.route("/")
def Home():
    return render_template("index.html")
#make predict route
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The Predicted Crop is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)