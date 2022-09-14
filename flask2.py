from flask import Flask, abort, request
import signal
import threading
import time
import os
import requests
app = Flask(__name__)
shuttingDown = False


def exit_call():
    time.sleep(20)
    requests.post("http://localhost:5420/_shutdown")
    # os._exit(0)


def exit_gracefully(self, signum):
    app.logger.error('Received shutdown signal. Exiting gracefully')
    global shuttingDown
    shuttingDown = True
    # TODO: wait for some time here to ensure we are not receiving any more
    # traffic
    _et = threading.Thread(target=exit_call)
    _et.daemon = True
    _et.start()


signal.signal(signal.SIGTERM, exit_gracefully)


@app.route("/")
def hello():
        return "Hello World!"


@app.route("/_status/liveness")
def liveness():
        return "I am alive"


@app.route("/_shutdown", methods=["POST"])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return "Not a werkzeug server"
    func()
    return "shutdown"


@app.route("/_status/readiness")
def readiness():
        if not shuttingDown:
            return "I am ready"
        else:
            abort(500, 'not ready anymore')


app.run(port=5420)