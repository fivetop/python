from bottle import route, run
import pymongo

@route('/hello1')
def hello1():
    connection = pymongo.MongoClient("localhost", 27017)
    db = connection.m101

    names = db.people

    item = names.find_one()

    item = names.find_one()
    return '<b>Hello %s!</b>' % item['name']



@route('/hello')
def hello():
    return "111Hello World!"


@route('/')
def index():
    return "ready!!!"


if __name__ == '__main__':
    run(host='localhost', port=8080, debug=True)