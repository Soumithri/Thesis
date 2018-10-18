import pymongo
from pprint import pprint
import tweepy
import time
import logging
import sys
import json, codecs
import timeit
import os
#import custom libraries
from MongoConnector import MongoConnector

config1 = {  'MONGO_COLL': 'social_coll',
            'MONGO_DB': 'tweetCorpus',
            'MONGO_HOST': 'localhost',
            'MONGO_PORT': 27017}
cursor2 = MongoConnector(config1).__connect__()
coll = config1['MONGO_COLL']

##########   TWITTER API ACCESS KEYS AND TOKENS #############
ACCESS_TOKEN_fol = "****"
ACCESS_TOKEN_SECRET_fol = "****"
CONSUMER_KEY_fol = "****"
CONSUMER_SECRET_fol = "****"

ACCESS_TOKEN_fr = "****"
ACCESS_TOKEN_SECRET_fr = "****"
CONSUMER_KEY_fr = "****"
CONSUMER_SECRET_fr = "****"

#############################################################

OUTPUT_DIRECTORY = './output'
SOCIAL_DIR = './output/social'
# fol_list = open(OUTPUT_DIRECTORY+"/followers_list",'w')
log_file = OUTPUT_DIRECTORY+"/social_info_log.log"
# Start logging
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')

def authenticate(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET):
    logging.debug("Creating the twitter authorization handler and connecting to the twitter...")
    try:
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    except tweepy.error.TweepError as E:
        logging.error("Error occured!!! Details: {0}\n".format(E))
        return -1
    else:
        logging.debug("Successfully authenticated OAUTH to twitter...")
        return api

def get_follower_data(api, user):
    try:
        logging.debug('Fetching followers_list for userid: {0}...'.format(user))
        followers_list = api.followers(id=user)
    except tweepy.error.TweepError as E:
        logging.error('Exception occurred for userid: {0}. Error Details --> {1}\n'.format(user, E))
        with codecs.open(OUTPUT_DIRECTORY+'/discarded_users_for_followers.txt', 'a', 'utf8') as f:
            f.write(user+'\n')
        return []
    except tweepy.error.RateLimitError as E:
        print(E)
        print('Rate Limit Reached!!! Sleeping for 15 mins...')  
    else:
        logging.debug('Total followers for userid: {0} --> {1}'.format(user, len(list(followers_list))))        
        return list(follower._json for follower in followers_list)

def get_friend_data(api, user):
    try:
        logging.debug('Fetching friends_list for userid: {0}...'.format(user))
        friends_list = api.followers(id=user)
    except tweepy.error.TweepError as E:
        logging.error('Exception occurred for userid: {0}. Error Details --> {1}\n'.format(user, E))
        with codecs.open(OUTPUT_DIRECTORY+'/discarded_users_for_friends.txt', 'a', 'utf8') as f:
            f.write(user+'\n')
        return []
    except tweepy.error.RateLimitError as E:
        print(E)
        print('Rate Limit Reached!!! Sleeping for 15 mins...')  
    else:
        logging.debug('Total friends for userid: {0} --> {1}'.format(user, len(list(friends_list))))        
        return list(friend._json for friend in friends_list)

def ensure_directory():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    if not os.path.exists(SOCIAL_DIR):
        os.makedirs(SOCIAL_DIR)

def main():
    # Start the timer
    start_time = timeit.default_timer()
    # Load the unique users from the file into a list given by unique_users_list\n",
    unique_users_list = list()
    logging.debug("Importing the unique users list.....")
    with open("./output/unique_users.txt",'r', encoding='utf-8') as outfile:
            unique_users_list = outfile.read().splitlines()
    logging.debug("Successfully imported {0} unique users....\n".format(len(unique_users_list)))

    # Creating the handler for twitter Authorization
    logging.debug("Creating the twitter authorization handler for followers and connecting to the twitter...")

    # ensure output directories are present
    ensure_directory()
    
    ##### FOLLOWER OAUTH HANDLER ######
    api_fol = authenticate(CONSUMER_KEY_fol, CONSUMER_SECRET_fol, ACCESS_TOKEN_fol, ACCESS_TOKEN_SECRET_fol)

    if api_fol== -1:
        logging.info("OUATH FAILURE (follower): Program is terminating..")
        return 1
    ###### FRIEND OAUTH HANDLER #############
    api_fr = authenticate(CONSUMER_KEY_fr, CONSUMER_SECRET_fr, ACCESS_TOKEN_fr, ACCESS_TOKEN_SECRET_fr)
    if api_fr== -1:
        logging.info("OUATH FAILURE (friend): Program is terminating..")
        return 1

    logging.info("---------------- COLLECTING FOLLOWERS & FRIENDS ----------------------\n\n")
    count = 0
    for user in unique_users_list:
        count+=1
        logging.debug("#### COUNT {0}... Retrieving social info for userid: {1}".format((count), user))
        ########## RETRIEVE FOLLOWERS ################             
        fol_list = get_follower_data(api_fol, user)
        
        ######## RETRIEVE FRIENDS ###################
        fr_list = get_friend_data(api_fr, user)
    
        social_dict = {'user_id': user,
                        'social_info': {
                                        'followers': fol_list,
                                        'friends': fr_list}}
        logging.debug("Saving the social info for userid: {0} in MongoCollection...".format(user))
        #pprint(follower_dict)
        with codecs.open(SOCIAL_DIR+'/userid_{}.json'.format(user), 'w', 'utf8') as f:
                f.write(json.dumps(social_dict))
        cursor2.insert_one(social_dict)
        logging.info("Successfully saved the social info info for userid: {0}...\n\n".format(user))

    end_time = timeit.default_timer()

    logging.debug("----> Program start time: {0}".format(start_time))
    logging.debug("----> Program end time: {0}".format(end_time))
    logging.debug("----> Total program time taken: {0} secs..".format(end_time - start_time))
        

if __name__ == '__main__':
    sys.exit(main())