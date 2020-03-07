from pymongo import MongoClient
import time
from pprint import pprint

client = MongoClient('mongodb://localhost:27017/')
db = client['tvshow_tweets']
stream_collection = db['streaming_coll']
hist_collection = db['historical_coll']
stream_collection_one_day = db['streaming_coll_1000']
hist_collection_one_day = db['historical_coll_1000']


def get_unique_users(collection):
    return collection.distinct('user.id_str')


def read_unique_users(file):

    with open(file, 'r') as infile:
        unique_users = (l.strip() for l in file)
        print(unique_users)
    return unique_users


if __name__ == '__main__':
    unique_users = get_unique_users(hist_collection)
    length = len(list(unique_users))
    print(length)
    for count, user in enumerate(unique_users):
        with client:
            cursor = stream_collection.find({'user.id_str': user})
            for i in cursor:
                stream_collection_one_day.insert_one(i)
           # stream_collection_one_day.insert_many(list(cursor))
            cursor2 = hist_collection.find({'user.id_str': user})
            for j in cursor2:
                hist_collection_one_day.insert_one(j)
           # hist_collection_one_day.insert_many(list(cursor2))
        if count % 100 == 0:
            print('Done storing for users: {}/{}'.format(count, length))
        if count == 1000:
            break
