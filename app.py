from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

with open("model/model.pkl", "rb") as m:
    model = pickle.load(m)

dataset = pd.read_csv("model/dataset.csv")
symptom_list = list(dataset.columns[1:])

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        selected_symptoms = request.form.get('selected_symptoms', '').split(',')
        selected_symptoms = [s.strip() for s in selected_symptoms if s.strip()]

        if not selected_symptoms:
            error = "Please select at least one symptom."

        else:
            input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
            input_vector = np.array(input_vector).reshape(1, -1)
            prediction = model.predict(input_vector)[0]

    return render_template('index.html', symptoms=symptom_list, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run()