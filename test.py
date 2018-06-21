import pandas as pd
import pickle
from flask import Flask, jsonify, request

### this is the post stuff###
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv('./data/iris.csv', names=names)
df = df.head(5)

"""Converting Pandas Dataframe to json """
data = df.to_json(orient='records')

#### PASS TO API #####
#### this is the flask part ###
test = pd.read_json(data, orient='records')
test = test[['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']]
score = test.drop(columns=['class'])

#clf = 'iris_knn_123.sav'
#clf = 'iris_svm_456.sav'
clf = 'iris_car_789.sav'
loaded_model = None
with open('./model/'+clf,'rb') as f:
  loaded_model = pickle.load(f)

cols = ['actual','prediction']
predictions = loaded_model.predict(score)
prediction_series = list(pd.Series(predictions))
final_predictions = pd.DataFrame(list(zip(test['class'], prediction_series)),columns=cols)
print final_predictions
#responses = jsonify(predictions=final_predictions.to_json(orient="records"))
#responses.status_code = 200

