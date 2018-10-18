import sys
import os
import logging
import timeit
import json
from bson import json_util

import pymongo
from pprint import pprint
import tweepy

from MongoConnector import MongoConnector

MONGO_CONFIG  = {  
    'MONGO_COLL': 'raw1Corpus',
    'MONGO_DB': 'tweetCorpus',
    'MONGO_HOST': 'localhost',
    'MONGO_PORT': 27017
    }
F_NAME = '/retweets_data2.json'
OUTPUT_DIRECTORY = './output'
log_file = OUTPUT_DIRECTORY+"/retweet_log.log"
# Start logging
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')


class Get_Retweets(object):

    def __init__(self, mongo_config=MONGO_CONFIG):
        logging.debug("Establishing MongoDB connection with parameters:- {}".format(mongo_config))
        self.cursor = MongoConnector(mongo_config).__connect__()
        logging.info("Established connection...\n")

    def get_retweets(self):
        '''Function that returns the retweets of each tweet collected 
           and stores it in a JSON format
        '''
        tweets_read = 0
        logging.debug("Checking for retweets...")
        with open(OUTPUT_DIRECTORY+F_NAME,'w') as f_out:
            total_tweets = self.cursor.find().count()
            logging.debug("Total Number of tweets (historical tweets) : {0}".format(total_tweets))
            for tweet in self.cursor.find():
                if tweet.get('retweeted_status'):
                    json.dump(tweet, f_out, default=json_util.default)
                    f_out.write('\n')
                    tweets_read += 1
            logging.debug("Total Number of retweets (historical tweets) : {0}".format(tweets_read))
            logging.debug("Total Number of non-retweets (historical tweets) : {0}".format(total_tweets-tweets_read))
        logging.debug("Succesully checked for retweets")


if __name__ == "__main__":
    # Start the timer
    start_time = timeit.default_timer()
    with open("db_config.json",'r') as f:
        config = json.load(f)
    retweets = Get_Retweets(config)
    retweets.get_retweets()
    end_time = timeit.default_timer()
    logging.debug("\n\n----> Program start time: {0}".format(start_time))
    logging.debug("----> Program end time: {0}".format(end_time))
    logging.debug("----> Total program time taken: {0} secs..".format(end_time - start_time))