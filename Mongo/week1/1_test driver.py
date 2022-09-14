import pymongo

connection = pymongo.MongoClient("localhost", 27017)
db = connection.m101

names = db.people

item = names.find_one()

print (item)
