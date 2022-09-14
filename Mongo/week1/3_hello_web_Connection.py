import bottle
#import pymongo

@bottle.route('/hello')
def hello():
    return "Hello World!"

@bottle.route('/index')
def index():
    from pymongo import Connection
    connection = Connection('localhost',27017)
    db = connection.m101
    names = db.people
    item = names.find_one()
    return '<b>Hello %s!</b>' % item['name']

bottle.run(host='localhost', port=8082)
# test URL: http://localhost:8082/
