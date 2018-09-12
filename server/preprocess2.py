#import the necessary libraries
import json
import sys
import codecs
from pprint import pprint
from string import punctuation
import timeit
import re
import logging
# import necessary NLTK packages
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
#import custom libraries
from MongoConnector import MongoConnector
from PyContract import PyContract

OUTPUT_DIRECTORY = './output_final'
config = {  'MONGO_COLL': 'raw1Corpus',
            'MONGO_DB': 'tweetCorpus',
            'MONGO_HOST': 'localhost',
            'MONGO_PORT': 27017}

customWords = ['bc', 'http', 'https', 'co', 'com','rt', 'one', 'us', 'new', 
              'lol', 'may', 'get', 'want', 'like', 'love', 'no', 'thank', 'would', 'thanks',
              'good', 'much', 'low', 'roger', 'im']
alphabets = list(map(chr, range(97, 123)))
myStopWords = set(stopwords.words('english') + list(punctuation) + customWords + alphabets)

# Initialize dbconnector, contracters, tokenizers, lemmatizers
dbconnector = MongoConnector(config)
contracter = PyContract()
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
lemmatizer = WordNetLemmatizer()

log_file = OUTPUT_DIRECTORY+"/pre_processing_log.log"
# Start logging
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')

def get_Tweets(user:str) -> dict:
    '''
        This function takes the config file and connects to MongoDB collection.
        Retrieves the tweet list from the user id and returns a dict object

        Output: {'user_id' : [tweet_list]}
    '''
    # Create new mongo collection and cursor object to store the unprocessed raw feature corpus
    cursor = dbconnector.__connect__()
    # Collect all the user tweets as one document and store it in a list
    que = cursor.find({'doc.user_info.id_str': user}, {"doc.tweets": 1})
    tweet_list = list()
    for i in que:
        for test in i['doc']['tweets']:
            tweet_list.append(contracter.__translate__(test['text']))
    return tweet_list

def preprocess_Tweets(tweet_list:list) -> list:
    # Pre-process step 1 - Word Tokenization
    
    # 1. Word Tokenization
    words = list(tokenizer.tokenize(tweets) for tweets in tweet_list)
    logging.debug("--------> Tokenization complete...")

    # 2. Remove the stop words from the document
    words_steps2 = list()
    for tweet in words: 
        sents = list(re.sub(r'\W+', '', word) for word in tweet)
        sents = filter(lambda s: not str(s).lstrip('-').isdigit(), sents)   
        sents = list(word for word in sents if word not in myStopWords and word!='' and 
                                                                    not word.startswith('http'))
        if sents!= None:
            words_steps2.append(sents)
    logging.debug("--------> Stop words removed...")

    # Pre-process step3 - Lemmatization
    pre_processed_list = list()
    for tweet in words_steps2:
        words_step4 = list()
        words_step3 = pos_tag(tweet)
        for token in words_step3:
            pos = get_wordnet_pos(token[1])
            # if verb, noun, adj or adverb include them after lemmatization
            if pos is not None and len(token[0]) > 3:
                try:
                    tok = lemmatizer.lemmatize(token[0], pos)
                    words_step4.append(tok)              
                except UnicodeDecodeError:
                    pass
        if(words_step4 != [] and words_step4!='\n'): 
            pre_processed_list.append(" ".join(words_step4))
        else:
            continue
    logging.debug("--------> Lemmatization complete...")
    return pre_processed_list


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return None

def ensure_directory():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

############    MAIN  PROGRAM STARTS HERE  #########

if __name__ == '__main__':

    # Start the timer
    start_time = timeit.default_timer()
    logging.debug("---------------- PROGRAM STARTING ----------------------\n")
    # Make sure directory is present. If not create the directory - 'output'
    with codecs.open(OUTPUT_DIRECTORY+"/raw_tweets_with_userid.csv", 'w','utf-8') as raw_u_file, \
           codecs.open(OUTPUT_DIRECTORY+"/preprocessed_tweets_corpora", 'w','utf-8') as preproc_file, \
             codecs.open(OUTPUT_DIRECTORY+"/preprocessed_tweets_with_userid.csv", 'w','utf-8') as preproc_u_file, \
               codecs.open(OUTPUT_DIRECTORY+"/preprocessed_tweets_1by1__with_userid", 'w','utf-8') as preproc_line_file:

        # Load the unique users from the file into a list given by unique_users_list
        logging.debug("Importing the unique users list.....")
        unique_users_list = list()
        with open(OUTPUT_DIRECTORY+"/final_unique_users.txt",'r', encoding='utf-8') as outfile:
            unique_users_list = outfile.read().splitlines()
        logging.debug("Successfully imported {0} unique users....\n".format(len(unique_users_list)))

        logging.debug("---------------- STARTING PRE-PROCESSING ----------------------\n\n")
        # Get tweets for each unique user
        counter = 1
        total_tweets = 0
        total_pre_processed_tweets = 0
        for user in unique_users_list:
            user_start_time = timeit.default_timer()
            logging.debug("{0}. Ready to process the tweets for userid: {1}".format(counter, user))
            logging.debug("----> Collecting tweets...")
            tweet_list = get_Tweets(user)
            logging.debug("----> Collected tweets: {0}...".format(len(tweet_list)))
            logging.debug("----> Pre-processing... ")
            processed_tweet_list = preprocess_Tweets(tweet_list)
            logging.debug("----> Pre-processing complete...")
            
            logging.debug("----> Writing files...")
            # Writing files
            print("{0}|{1}".format(user, tweet_list), file=raw_u_file)
            print("{0}|{1}".format(user, processed_tweet_list), file=preproc_u_file)
            for tweet in processed_tweet_list:
                print("{0}|{1}".format(user, tweet), file=preproc_line_file)
                print("{0}".format(tweet), file=preproc_file, end=" ")          
            logging.debug("----> Completed writing files...")
            user_end_time = timeit.default_timer()
            logging.debug("############ SUMMARY ###########")
            logging.debug("----> No. of tweets before pre-processing: {0}".format(len(tweet_list)))
            logging.debug("----> No. of tweets after pre-processing: {0}".format(len(processed_tweet_list)))
            logging.debug("----> No. of tweets discarded: {0}".format(len(tweet_list)-len(processed_tweet_list)))
            logging.debug("----> Total time taken: {0}".format(user_end_time-user_start_time))
            logging.debug("################################\n")   
            total_tweets += len(tweet_list)
            total_pre_processed_tweets += len(processed_tweet_list)
            print('{0}. Pre-processed tweets for userid: {1}'.format(counter, user))
            counter+=1
     # Print the time taken     
    end_time = timeit.default_timer()
    logging.debug("---------------- PRE-PROCESSING COMPLETED----------------------\n\n")
    logging.debug("############ PROGRAM SUMMARY ###########")
    logging.debug("----> No. of unique users: {0}".format("2207"))
    logging.debug("----> No. of discarded users: {0}".format("40"))
    logging.debug("----> No. of pre-processed users: {0}".format(len(unique_users_list)))
    logging.debug("----> No. of tweets before pre-processing: {0}".format(total_tweets))
    logging.debug("----> No. of tweets before pre-processing: {0}".format(total_pre_processed_tweets))
    logging.debug("----> No. of discarded tweets for pre-processing (not including discared users): \
                                                {0}".format(total_tweets - total_pre_processed_tweets))
    logging.debug("----> Program start time: {0}".format(start_time))
    logging.debug("----> Program end time: {0}".format(end_time))
    logging.debug("----> Total program time taken: {0}".format(end_time - start_time))
    logging.debug("################################\n\n")
    logging.debug("---------------- PROGRAM ENDED ----------------------")
###########     MAIN  PROGRAM ENDS HERE    ##########ß