# -*- coding: UTF-8 -*-
import tweepy
import settings
import tweepy
import json
from pymongo import MongoClient
import logging
from datetime import datetime
import timeit
from pprint import pprint
from tweepy.error import TweepError

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

log_file = 'historical_tweet_collection.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

client = MongoClient('mongodb://localhost:27017/')
db = client['tvshow_tweets']
collection = db['streaming_coll']
hist_collection = db['historical_coll']

failed_users = []


def get_unique_stream_users():
    return collection.distinct('user.screen_name')


def get_unique_historical_users():
    return hist_collection.distinct('user.screen_name')


def get_tweets(user_screen_name):
    count = 0
    cursor = tweepy.Cursor(api.user_timeline,
                           screen_name=user_screen_name).items()
    try:
        for status in cursor:
            data_to_insert = dict(status._json)
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
                hist_collection.insert_one(data_to_insert)
            except Exception as e:
                logging.exception(e)
            else:
                count += 1
    except TweepError as e:
        failed_users.append(user_screen_name)
        logging.exception(e)
        logging.info('Collected number of historical tweets: {}'.format(count))
        pass
    return count


start_time = timeit.default_timer()
logging.info("----> Program started at : {0}".format(datetime.now()))
auth = tweepy.OAuthHandler(settings.TWITTER_KEY, settings.TWITTER_SECRET)
auth.set_access_token(settings.TWITTER_APP_KEY,
                      settings.TWITTER_APP_SECRET)
api = tweepy.API(auth,
                 wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)
unique_stream_user_list = list(get_unique_stream_users())
unique_hist_user_list = list(get_unique_historical_users())
screen_names = [i for i in unique_stream_user_list if i not in unique_hist_user_list]
logging.info('Number of unique streaming users: {}'.format(len(screen_names)))
total_count = 0
for user_screen_name in screen_names:
    logging.info('Starting to collect historical tweets for user: {}'.format(user_screen_name))
    count = get_tweets(user_screen_name)
    total_count += count
    logging.info('Finished collected historical tweets for user: {}'.format(user_screen_name))

with open('failed_users.txt', 'w') as file:
    file.write(failed_users)
logging.info('Total collected number of historical tweets: {}'.format(total_count))
end_time = timeit.default_timer()
logging.info("----> Program start time: {0}".format(start_time))
logging.info("----> Program end time: {0}".format(end_time))
logging.info("----> Total program time taken: {0} secs..".format(end_time - start_time))
logging.info("----> Program ended at : {0}".format(datetime.now()))
