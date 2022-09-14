import pymongo
connection = pymongo.MongoClient("localhost", 27017)
db = connection.symbols
collection  = db.symbols
collection.insert({"number":0})


collection.update({"number":0}, {"number":9999})

collectionInfo = db.collection_names()

print(collectionInfo)

# for i in range(0, 100):
#  collection.insert({"student_id":i})



docs = collection.find()

for i in docs:
  print(i)


