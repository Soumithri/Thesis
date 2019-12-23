#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:48:51 2019

@author: soumithri
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pymongo import MongoClient
import networkx as nx
import logging
from .Tweet import TweetType, Tweet
from .input import NODE_GRAPH


class TweetsNetwork:

    # Vertex attribute
    USER_CREATE_TIME    = "create_time"
    #DB_NAME1 = "old_tweets_2017"
    BATCH_SIZE = 100

    # Edge attribute
    EDGE_CREATE_TIME    = "create_time"
    EDGE_RETWEET_COUNT  = "retweet_count"
    EDGE_QUOTE_COUNT    = "quote_count"
    EDGE_REPLY_COUNT    = "reply_count"
    EDGE_MENTION_COUNT  = "mention_count"

    UNIQUE_USERS_FILE  = './final_unique_users.txt'
    
    TOP100_USERS = './top100users.csv'
    BOT100_USERS = './bot100users.csv'

    def __init__(self, show, coll, DB_NAME):
        logging.basicConfig(filename= show + ".log", level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info("starting...")
        self.network = nx.DiGraph()
        self.db = MongoClient()[DB_NAME]
        self.coll = self.db[coll]
        #print(self.coll.find_one())
        self.show = show
        self.PATTERN = "^" + show

    def add_user(self, user_id, create_time):
        assert isinstance(user_id, int), "User id must be int!"
        if user_id not in self.network:
            self.network.add_node(user_id, create_time = create_time)
            logging.info('Added user_id: {0} to the network'.format(user_id))
            #print(self.network.node[user_id])
        else:
            #print('already_present')
            if create_time < self.network.node[user_id][self.USER_CREATE_TIME]:
                self.network.node[user_id][self.USER_CREATE_TIME] = create_time
            #print('adjusted...')

    def get_tweet_type_count(self, tweet_type):
        if tweet_type is TweetType.RETWEET:
            return self.EDGE_RETWEET_COUNT
        elif tweet_type is TweetType.QUOTE:
            return self.EDGE_QUOTE_COUNT
        elif tweet_type is TweetType.REPLY:
            return self.EDGE_REPLY_COUNT
        elif tweet_type is TweetType.MENTION:
            return self.EDGE_MENTION_COUNT
        else:
            assert False, "Unknown tweet type!"

    def add_edge(self, start, end, tweet_type, create_time):
        # print(self.network.nodes)
        # print(start, end, type(start), type(end))
        # assert self.network.has_node(start)!=True, 'start is not present'
        # assert self.network.has_node(str(end))!=True, 'end is not present'
        assert start in self.network and end in self.network, "User id does not exist!"
        assert isinstance(tweet_type, TweetType), "Tweet type must be instance of TweetType"

        # Add edge
        if self.network.has_edge(start, end):
            if create_time < self.network[start][end][TweetsNetwork.EDGE_CREATE_TIME]:
                self.network[start][end][TweetsNetwork.EDGE_CREATE_TIME] = create_time
        else:
            #tmp = TweetsNetwork.EDGE_CREATE_TIME
            self.network.add_edge(start, end,create_time = create_time)

        # Count edge type
        edge_type_count = self.get_tweet_type_count(tweet_type)
        if edge_type_count not in self.network[start][end]:
            self.network[start][end][edge_type_count] = 1
        else:
            self.network[start][end][edge_type_count] += 1

    def add_retweet_edge(self, tweet):
        if tweet.retweet_author_id not in self.network:
            #self.add_user(tweet.retweet_author_id, create_time = tweet.retweet_create_time)
            #logging.info('Added retweeted author id: {} to the network'.format(tweet.retweet_author_id))
            return
        self.add_edge(tweet.author_id, tweet.retweet_author_id, TweetType.RETWEET, tweet.create_time)
        logging.info('added retweet edge...')

    def add_quote_edge(self, tweet):
        if tweet.quote_author_id not in self.network:
            return
        self.add_edge(tweet.author_id, tweet.quote_author_id, TweetType.QUOTE, tweet.create_time)

    def add_mention_edge(self, tweet):
        for mentioned_user in tweet.mentions:
            if (tweet.retweet and mentioned_user in tweet.retweet_mentions) or \
                    (tweet.quote and mentioned_user in tweet.quote_mentions) or \
                    mentioned_user not in self.network:
                continue


            self.add_edge(tweet.author_id, mentioned_user, TweetType.MENTION, tweet.create_time)

    def add_reply_edge(self, tweet):
        if tweet.reply_id not in self.network:
            return
        self.add_edge(tweet.author_id, tweet.reply_id, TweetType.REPLY, tweet.create_time)

    def add_tweet(self, tweet):
        # Retweet do not have any new content, retweet with comment is called quote.
        # Note: It is possible that A retweet a quote, then tweet will contain both retweet_status and quote_status
        # Vice Versa.

        #logging.info('in add_tweet.. function')
        #print(tweet.retweet)
        if tweet.retweet:
            self.add_retweet_edge(tweet)
            logging.debug('tweet is a retweet')
            
        if tweet.quote:
            self.add_quote_edge(tweet)

        if tweet.mentions:
            self.add_mention_edge(tweet)

        if tweet.reply_id:
            self.add_reply_edge(tweet)

    def add_show(self, show_name):
        """
        Edge: A retweeted B(A follows B)
        A -> B

        Edge: A mentioned B
        A -> B

        If A follows B, A is B's follower, B is A's friend.

        Graph format:
        Node ID: user id in long format
        Node Attribute:
            'active_timestamp':     The first time user post a tweet related to the tv show
        Edge Attribute "Type":
            'R':                    Retweet edge
            'M':                    Mention edge
            'Q':                    Quote edge
            'C':                    Reply edge
            "RMQ":                  Retweet and Mention and Quote edge

        """
        assert show_name != '', "Hashtag shouldn't be empty!"

        # Added tweet author to network
        query_string = {
                        'entities.hashtags.text':{'$regex': self.PATTERN, '$options': 'i'}
                       }
        logging.info('\nQuery Tweets Begin\nQuery string is ' + str(query_string))
        for t in self.coll.find(query_string).batch_size(TweetsNetwork.BATCH_SIZE):
            #print(t['id_str'])
            tweet = Tweet(t)
            self.add_user(tweet.author_id, tweet.create_time)
        #print('added users')
        tweet_users = self.network.number_of_nodes()
        #print('getting_info')
        logging.info('Analysis tweets with hashtag {} finished. {} users added.'.format(show_name,tweet_users))

        # Added historical tweets
        count = 0
        for author_id in self.network.nodes():
            query_string = {"user.id": author_id}
            for t in self.coll.find(query_string).batch_size(TweetsNetwork.BATCH_SIZE):
                self.add_tweet(Tweet(t))
                #print('adding..: ',author_id )
            if count % 50 == 0:
                logging.info("Historical tweets: {} users done, {} users remain.".format(count, tweet_users-count))
            if count % 100 == 0:
                self.save()
            count += 1

        tweet_edges = self.network.number_of_edges()
        logging.info('Analysis historical tweets finished. {} edges added.'.format(tweet_edges))

    def save(self, filename=NODE_GRAPH):
        nx.write_graphml(self.network, filename)

    def get_user_list(self, file):
        with open(file, 'r') as infile:
            unique_users = infile.read().splitlines()
        return unique_users