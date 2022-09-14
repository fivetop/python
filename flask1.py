import signal
import threading
import time
import os
import requests
import time
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
from flask import Flask, render_template

sched = BackgroundScheduler()

app = Flask(__name__)

# 일반적인 라우트 방식입니다.
@app.route('/board')
def board():
    return "그냥 보드"

# URL 에 매개변수를 받아 진행하는 방식입니다.
@app.route('/board/<article_idx>')
def board_view(article_idx):
    return article_idx

# 위에 있는것이 Endpoint 역활을 해줍니다.
@app.route('/boards',defaults={'page':'index'})
@app.route('/boards/<page>')
def boards(page):
    return page+"페이지입니다."

@app.route('/mongo',methods=['GET', 'POST'])
def mongoTest():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.newDatabase
    collection = db.mongoTest
    results = collection.find()
    client.close()
    return render_template('mongo.html', data=results)

@app.route('/')
def hello_world():
    return 'Hello World!'

# 5초마다 실행
@sched.scheduled_job('interval', seconds=5, id='test_1')
def job1():
    print(f'job1 : {time.strftime("%H:%M:%S")}')
    requests.post("http://localhost:5005/mongo")

sched.start()
app.run(host="localhost",port=5005)