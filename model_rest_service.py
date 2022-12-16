from flask import Flask, request
import pickle
import numpy as np

local_classifier = pickle.load(open("classifier.pickle", 'rb'))
local_scaler = pickle.load(open("standard_scaler.pickle", 'rb'))

app = Flask(__name__)
@app.route("/model", methods = ["POST"])
def get_data():
    request_data = request.get_json(force = True)
    age = request_data["age"]
    salary = request_data["salary"]
    prediction = local_classifier.predict(np.array([[age, salary]]))
    return "Prediction : {}".format("Yes" if prediction == 1 else "No")
    
    
if __name__ == "__main__":
    app.run(port = 8000, debug = True)