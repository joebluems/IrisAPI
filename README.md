# IrisAPI

Building several models (offline):
> python iris_model.py

Create container...
Create environment... 
sudo pip install gunicorn
sudo pip install flask
sudo pip install sklearn
sudo pip install scipy

MODEL #1:
> gunicorn --bind 0.0.0.0:5000 server:app

MODEL #2:
> gunicorn --bind 0.0.0.0:5001 server2:app

MODEL #3:
> gunicorn --bind 0.0.0.0:5002 server3:app

SCORE:
> python post.py

