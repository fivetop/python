#import pymongo
import bottle
import pymongo
import sys

connection = pymongo.MongoClient("localhost", 27017)

##connection = Connection("localhost", 27017)
db = connection.m101
people = db.people


for i in range(1, 300000):
    person = {'No': 0, 'name': 'Barack Obama', 'role': 'President'}
    person['No'] = i
    people.insert(person)
    print(i)

print (person)

try:
    people.insert(person)
except:
    print ("insert failed:", sys.exc_info()[0],'--',sys.exc_info()[1])

