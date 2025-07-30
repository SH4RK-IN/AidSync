from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

dataset = pd.read_csv("model/dataset.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

model = RandomForestClassifier()

model.fit(X, y)

with open("model/model.pkl", "wb") as m:
    pickle.dump(model, m)