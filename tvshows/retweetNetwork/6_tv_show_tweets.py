#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:45:58 2019

@author: soumithri
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from retweetNetwork.tvshow_tweetsNetwork import TweetsNetwork
from retweetNetwork.Tweet import  Tweet
from retweetNetwork.NetworkUtils import get_graphml
import logging
import timeit
from retweetNetwork.input import TV_SHOW, DB_NAME, STREAM_COLL, HIST_COLL, NODE_GRAPH, LOG_FILE, DATA

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(message)s')

def main():

    start_time = timeit.default_timer()
    if DATA == 'HISTORICAL_DATA':
        network = TweetsNetwork(TV_SHOW, HIST_COLL, DB_NAME)
        try:
            network.network = get_graphml(NODE_GRAPH)
        except FileNotFoundError:
            logging.info('Creating the first graph file...')
        else:
            logging.info('Loading the graph file...')
            network.network = get_graphml(NODE_GRAPH)
        network.add_show(TV_SHOW)
        network.save()
           
    elif DATA == 'STREAM_DATA':
        network = TweetsNetwork(TV_SHOW, STREAM_COLL, DB_NAME)
        try:
            network.network = get_graphml(NODE_GRAPH)
        except FileNotFoundError:
            logging.info('graph file not found... please create the graph file using streaming tweets first..')
            sys.exit()
        else:
            network_nodes = [str(i) for i in network.network.nodes()]
            for count, author_id in enumerate(network.coll.distinct('user.id')):
                logging.info('Adding streaming tweets for authorid:{}'.format(author_id))
                if str(author_id) in network_nodes:
                    #print(author_id)
                    #print(network.coll.find_one())
                    query_string = {"user.id_str": str(author_id)}  ## node of networkx is string type
                    #print(network.coll.find_one(query_string))
                    for t in network.coll.find(query_string):
                        #print(t['id_str'])
                        #print(network)
                        #print(Tweet(t).retweet, Tweet(t).author_id, Tweet(t).retweet_author_id)
                        network.add_tweet(Tweet(t))
                    if count % 1000 == 0:
                        logging.info("Historical tweets: {} users done, {} users remain.".format(count, network.network.number_of_nodes()))
                    if count % 2000 == 0:
                        network.save()
                else:
                    logging.info('User id not present in the histoical tweets: {}'.format(author_id))
    end_time = timeit.default_timer()
    total_time = (end_time-start_time)/60
    logging.info("----> Total program time taken: {0} mins".format(total_time))


if __name__ == "__main__":
    main()
