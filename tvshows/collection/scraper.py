# -*- coding: UTF-8 -*-
import tweepy
import settings
import tweepy
import json
from pymongo import MongoClient
import logging
from datetime import datetime
import timeit

ENTITIES = 'entities'
EXTENDED_TWEET = 'extended_tweet'
FULL_TEXT = 'full_text'
HAS_MENTION = 'has_mention'
HAS_QOUTE = 'has_quote'
HAS_REPLY = 'has_reply'
IN_REPLY_TO_USER_ID = 'in_reply_to_user_id'
IS_QUOTE_STATUS = 'is_quote_status'
IS_RETWEET = 'is_retweet'
QUOTED_STATUS = 'QUOTED_STATUS'
RETWEETED_STATUS = 'retweeted_status'
TEXT = 'text'
USER_MENTIONS = 'user_mentions'

languages = ['en']
log_file = 'streaming_collection_one_day.log'
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')


client = MongoClient('mongodb://localhost:27017/')
db = client['tvshow_tweets_one_day']
collection = db['streaming_coll']
WORDS = ['#YouNetflix']


class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        data_to_insert = dict(status._json)
        if data_to_insert.get('lang') not in languages:
            return
        is_retweet = hasattr(status, RETWEETED_STATUS)
        has_mention = data_to_insert.get(ENTITIES, '').get(USER_MENTIONS) \
            is not None
        has_reply = data_to_insert.get(IN_REPLY_TO_USER_ID) is not None
        if hasattr(status, EXTENDED_TWEET):
            text = status.extended_tweet[FULL_TEXT]
        else:
            text = status.text
        data_to_insert[TEXT] = text
        data_to_insert[IS_RETWEET] = is_retweet
        data_to_insert[HAS_MENTION] = has_mention
        data_to_insert[HAS_QOUTE] = data_to_insert.get(IS_QUOTE_STATUS)
        data_to_insert[HAS_REPLY] = has_reply

        try:
            collection.insert_one(data_to_insert)
        except Exception as e:
            logging.exception(e)
        else:
            logging.debug('tweets written to db...')

    def on_error(self, status_code):
        if status_code == 420:
            return False


# Start the timer
start_time = timeit.default_timer()
logging.info("----> Program started at : {0}".format(datetime.now()))
auth = tweepy.OAuthHandler(settings.TWITTER_KEY, settings.TWITTER_SECRET)
auth.set_access_token(settings.TWITTER_APP_KEY, settings.TWITTER_APP_SECRET)
api = tweepy.API(auth)
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth,
                         wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True,
                         listener=myStreamListener)
myStream.filter(track=WORDS, languages=['en'])
end_time = timeit.default_timer()
logging.info("----> Program start time: {0}".format(start_time))
logging.info("----> Program end time: {0}".format(end_time))
logging.info("----> Total program time taken: {0} secs..".format(end_time - start_time))
logging.info("----> Program ended at : {0}".format(datetime.now()))
