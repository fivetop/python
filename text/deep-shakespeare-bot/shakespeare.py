import tweepy
import time
import config

from past.builtins import execfile

FILE_PATH = 'gen-queue.txt'

auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)

auth.set_access_token(config.access_token, config.access_token_secret)

api = tweepy.API(auth)

queue_text = []


def generate_new():
    execfile("generate.py")
    with open(FILE_PATH, "r") as queue:
        text = queue.read()
        global queue_text
        queue_text = text.split("%")


def post():
    if len(queue_text) == 0:
        generate_new()
    api.update_status(queue_text[0])
    queue_text.pop(0)


while True:
    post()
    time.sleep(10)
