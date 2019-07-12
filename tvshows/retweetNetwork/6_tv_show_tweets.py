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
from retweetNetwork.NetworkUtils import get_graphml
import logging
import timeit
from retweetNetwork.input import TV_SHOW, DB_NAME, NODE_GRAPH, LOG_FILE


TV_SHOW = TV_SHOW
FILE = "Tweets"
TWEET_COLL = DB_NAME
OLD_COLL = "stream_tweets"
LOG_FILE = LOG_FILE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(message)s')


def main():
    # Start the timer
    start_time = timeit.default_timer()
    network = TweetsNetwork(TV_SHOW, DB_NAME)
    try:
        network.network = get_graphml(NODE_GRAPH)
    except FileNotFoundError:
        logging.info('Creating the first graph file...')
    else:
        logging.info('Loading the graph file...')
        network.network = get_graphml(NODE_GRAPH)
    network.add_show(TV_SHOW)
    network.save()
    end_time = timeit.default_timer()
    total_time = (end_time-start_time)/60
    logging.info("----> Total program time taken: {0} mins".format(total_time))


if __name__ == "__main__":
    main()
