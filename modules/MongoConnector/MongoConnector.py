#import modules
from pymongo import MongoClient
import json

# Helper class for connecting mongodb using a dict object named 'config'
class MongoConnector:
    '''
        Class used to connect mongodb usind a parameter - "config" which is a dict object.
        Usage:  >> config = {
                                "MONGO_HOST" : "localhost", 
                                "MONGO_PORT" : 27017,
                                "MONGO_DB"   : "tweetCorpus",
                                "MONGO_COLL" : "historical_tweets2"
                            }
                >> MongoCursor = MongoConnector(config).__connect__()
    '''

    def __init__(self, config:dict) -> None:
        ''' 
            Initializes mongodb environment variables...  
            It contains a parameter 'config' which is a dictionary object containing the 
            parameters - "MONGO_HOST", "MONGO_PORT", "MONGO_DB", "MONGO_COLL"
        ''' 
        self.config = config

    def __enter__(self):
        print("Connecting to the MongoDB...")

    def __connect__(self)-> 'cursor':
        ''' 
            Establishes mongodb connection from the instance of the MongoConnector class
            ans returns the cursor object...
            Usage: >> db.__connect__()  #where db is the instance of the MongoConnector class 
        '''
        self.client = MongoClient(self.config["MONGO_HOST"], self.config["MONGO_PORT"])
        self.db = self.client[self.config["MONGO_DB"]]
        self.coll = self.db[self.config["MONGO_COLL"]]
        return self.coll

    def __exit__(self, exception_type, exception_value, traceback):
        self.client.close()
