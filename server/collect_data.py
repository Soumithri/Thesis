from __future__ import print_function
import tweepy
import json
from pymongo import MongoClient
import logging
from datetime import datetime

MONGO_HOST= 'mongodb://localhost' 
WORDS = ['']

CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

log_file = 'streaming_collection.log'
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')

class StreamListener(tweepy.StreamListener):    
    #This is a class provided by tweepy to access the Twitter Streaming API. 

    def on_connect(self):
        # Called initially to connect to the Streaming API
        print("You are now connected to the streaming API.")
 
    def on_error(self, status_code):
        # On error - if an error occurs, display the error / status code
        print('An Error has occured: ' + repr(status_code))
        return False
 
    def on_data(self, data):
        #This is the meat of the script...it connects to your mongoDB and stores the tweet
        try:
            client = MongoClient(MONGO_HOST)
            
            # Use twitterdb database. If it doesn't exist, it will be created.
            db = client.tweetCorpus
    
            # Decode the JSON from Twitter
            datajson = json.loads(data)
            
            #grab the 'created_at' data from the Tweet to use for display
            created_at = datajson['created_at']

            #print out a message to the screen that we have collected a tweet
            print("Tweet collected at " + str(created_at))
            
            #insert the data into the mongoDB into a collection called twitter_search
            #if twitter_search doesn't exist, it will be created.
            db.streaming_coll.insert(datajson)
        except Exception as e:
           logging.exception(e)

# Start the timer
start_time = timeit.default_timer()
logging.info("----> Program started at : {0}".format(datetime.now()))
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
#Set up the listener. The 'wait_on_rate_limit=True' is needed to help with Twitter API rate limiting.
listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True)) 
streamer = tweepy.Stream(auth=auth, listener=listener)
print("Tracking: " + str(WORDS))
streamer.filter(track=WORDS, languages='en')

end_time = timeit.default_timer()
logging.info("----> Program start time: {0}".format(start_time))
logging.info("----> Program end time: {0}".format(end_time))
logging.info("----> Total program time taken: {0} secs..".format(end_time - start_time))
logging.info("----> Program ended at : {0}".format(datetime.now()))
