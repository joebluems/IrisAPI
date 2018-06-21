import pandas as pd
import json
import requests

"""Setting the headers to send and accept json responses """
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

"""Reading test batch """
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv('./data/iris.csv', names=names)
df = df.head()

"""Converting Pandas Dataframe to json """
data = df.to_json(orient='records')

"""POST <url>/predict """
resp = requests.post("http://0.0.0.0:5000/predict", data = json.dumps(data),headers= header)

print resp.status_code
print resp.json()
