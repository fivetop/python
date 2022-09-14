
import pymongo
import sys

# establish a connection to the database
#connection = pymongo.MongoClient("mongodb://localhost", safe=True)
connection = pymongo.MongoClient("localhost", 27017)
# get a handle to the school database
db=connection.school
scores = db.scores

def find_one():

    print ("find one, reporting for duty")
    query = {'student_id':10}
    
    try:
        doc = scores.find_one(query)
        
    except:
        print ("Unexpected error:", sys.exc_info()[0])

    
    print (doc)


find_one()

