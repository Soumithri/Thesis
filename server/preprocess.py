#import the necessary libraries
import nltk
import json
import pymongo
import sys
from pprint import pprint

#import custom libraries
from MongoConnector import MongoConnector


# Load the config dictionary object from the db_config.json file
with open("db_config.json",'r') as f:
    config = json.load(f)
# print the config file
pprint(config)

# Create a cursor to connect to MongoDB 
cursor = MongoConnector(config).__connect__()

# Get a document from the mongodb collection
#documents = cursor.find({}).limit(1)
#pprint(list(documents))



# Load the unique users from the file into a list given by unique_users_list\n",
unique_users_list = list(),
with open("./output/unique_users.txt",'r', encoding='utf-8') as outfile:
        unique_users_list = outfile.read().splitlines()

# Collect the tweets of the unique users and save them in a dictionary
# Store the user info (id_str, screen_name) , tweet info (tweet_id, tweet, tweet_status, truncated),
# user mention info(mentions, id_str, names, screen-names), favorite info (favorited, favorite_count),
# retweet info (retweeted, retweet_count), reply (replyreply_id, reply_count), quote (quote_status, quote_list)

# Create new mongo collection and cursor object to store the unprocessed raw feature corpus
config1 = {  'MONGO_COLL': 'raw1Corpus',
            'MONGO_DB': 'tweetCorpus',
            'MONGO_HOST': 'localhost',
            'MONGO_PORT': 27017}
cursor2 = MongoConnector(config1).__connect__()

MAX_TWEET_LIMIT = 3000
# Create the structure of the dictionary
tweet_dict = dict()
user_visible_list = ['user.id_str',
                    'user.name', 
                    'user.screen_name',
                    'user.created_at',
                    'user.description',
                    'user.derived',
                    'user.protected',
                    'user.verified',
                    'user.followers_count',
                    'user.friends_count',
                    'user.listed_count',
                    'user.favourites_count',
                    'user.statuses_count',
                    'user.contributors_enabled']
tweet_visible_list = [  'id_str',
                        'created_at',
                        'text',
                        'tweet_status',
                        'truncated',
                        'entities.user_mentions',
                        'favorited',
                        'favorite_count',
                        'retweeted',
                        'retweet_count',
                        'in_reply_to_screen_name',
                        'in_reply_to_status_id_str',
                        'in_reply_to_user_id',
                        'reply_count',
                        'is_quote_status',
                        'quote_count']
user_projection = {attribute : 1 for attribute in user_visible_list}
tweet_projection = {attribute: 1 for attribute in tweet_visible_list}

discarded_users_list = list()
counter = 0

for user in unique_users_list:
    tweet_list = list()
    query = {'user.id_str' : user,  'lang': 'en'}
    
    tweet_count = cursor.find(query, user_projection).count()
    if (tweet_count<10):
        discarded_users_list.append(user)
        print('discarding userid : %s...' %(user))
        counter+=1
        continue

    user_docs = cursor.find_one(query, user_projection)
    tweet_docs = cursor.find(query, tweet_projection).limit(MAX_TWEET_LIMIT)
    tweet_list = list(tweet_docs)
    print("Storing raw info of user with userid : %s" %(user))

    # Discard users whose tweets < 10

    #pprint(list(tweet_docs))
    tweet_dict['doc'] = {
                        'user_info': user_docs['user'],
                        'tweets': tweet_list
                        }

    cursor2.insert_one(tweet_dict)
    tweet_dict.clear()        
    counter+=1
    print('Storing done for %d user with id: %s' %(counter, user))
