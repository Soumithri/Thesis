# import in-built python modules
import os
import sys
import logging
from datetime import datetime
from random import sample
#import third party modules
import codecs
from timeit import default_timer
from gensim import corpora, models
from argparse import ArgumentParser 

INPUT_DIR = '././models/'
OUTPUT_DIR = '././models/' + datetime.now().strftime('%m-%d-%Y %H:%M:%S')


def create_Directories(path):
        ''' Checks for the directory. If not present creates the directory'''
        try: 
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

def lda_train(corpus, id2word, args): 
    with codecs.open(args['output_directory']+'/perplexity.csv','w','utf-8') as f1, \
           codecs.open(args['output_directory']+'/topic_coherence.csv','w','utf-8') as f2:
        iter_range = args['num_topics_range'].split()
        for num_topics in range(int(iter_range[0]), int(iter_range[1]), int(iter_range[2])):
            print("Training LDA(k=%d)" % num_topics)
            lda = models.ldamulticore.LdaMulticore(corpus=corpus, id2word=id2word, chunksize=args['chunk_size'],
                                                    num_topics=num_topics, passes=args['passes'], 
                                                    iterations=args['iterations']
                                                    )   
            lda.save(args['output_directory']+'/lda_num_topics_k%d.lda' % num_topics, separately=False)
            f1.write(str(num_topics) + ',' + str(lda.log_perplexity(corpus)) + '\n')
            # Generate topic coherence model, save it and calculate topic coherence
            coh = models.coherencemodel.CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, 
                                                        coherence='u_mass')
            coh.save(args['output_directory']+'/coh_num_topics_k%d.lda' % num_topics, separately=False)                                            
            f2.write(str(num_topics) + ',' + str(coh.get_coherence()) + '\n')

def build_models(mm_corpus, dictionary, args):
    # Call the lda_train() method
    lda_models = lda_train(corpus=mm_corpus, id2word=dictionary, args=args) 

def load_dictionary(location):
    dictionary = corpora.Dictionary.load(location)
    return dictionary

def load_corpus(location):
    mm_corpus = corpora.MmCorpus(location)
    return mm_corpus

def main():
    # Creating the Argument Parser
    parser = ArgumentParser(description="Parser for building LDA models using GENSIM library...")
    parser.add_argument('-c','--corpus', type=str, help="Absolute path of the GENSIM MM_Corpus", required=True)
    parser.add_argument('-d','--dictionary',type =str, help="Absolute path of the GENSIM dictionary", 
                        required=True)
    parser.add_argument('-cs','--chunk_size', type=int, help="Chunk size to process", default=2000, nargs='?')
    parser.add_argument('-k','--num_topics_range', type=str, help="Number of Topics", default="10 101 10", nargs='?')
    parser.add_argument('-a','--alpha', help="Value of hyper-parameter: alpha", default='symmetric', nargs='?')
    parser.add_argument('-e','--eta', help="Value of hyper-parameter: eta", default=None, nargs='?')
    parser.add_argument('-p','--passes', type=int, help="Number of passes", default=1, nargs='?')
    parser.add_argument('-i','--iterations', type=int, help="Number of iterations", default=50, nargs='?')
    parser.add_argument('-o','--output_directory', type=str, 
                        help="Absolute path of the output directory to save models", default=OUTPUT_DIR, nargs='?')
    args = vars(parser.parse_args())

    #logging.info("Creating the output directory: {0}".format(args['output_directory']))
    create_Directories(args['output_directory'])
    #logging.info("Successfully created the directories...")
    log_file = args['output_directory'] + "/lda_gensim.log"
    logging.basicConfig(filename=log_file, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Start the timer
    start_time = default_timer()
    logging.info("---------------- PROGRAM STARTED----------------------\n\n")
    logging.info("---> Command line parameters:")
    logging.info("\t---> corpus: {0}".format(args['corpus']))
    logging.info("\t---> dictionary: {0}".format(args['dictionary']))
    logging.info("\t---> chunk size: {0}".format(args['chunk_size']))
    logging.info("\t---> num_topics_range: {0}".format(args['num_topics_range']))
    logging.info("\t---> alpha: {0}".format(args['alpha']))
    logging.info("\t---> eta: {0}".format(args['eta']))
    logging.info("\t---> passes: {0}".format(args['passes']))
    logging.info("\t---> iterations: {0}".format(args['iterations']))
    logging.info("\t---> output_directory: {0}".format(args['output_directory']))
    logging.info("\n\n")

    logging.info("loading Dictionary...")
    dictionary = load_dictionary(args['dictionary'])
    logging.info("Successfully loaded the Dictionary...")
    logging.info("loading MM Corpus...")
    mm_corpus = load_corpus(args['corpus'])
    logging.info("Successfully loaded the MM Corpus...\n")
    logging.info("Building LDA models using GENSIM...")
    build_models(mm_corpus, dictionary, args)
    logging.info("Successfully built the LDA models...\n\n")
    end_time = default_timer()

    logging.info("############ PROGRAM EXECUTION SUMMARY ###########")
    logging.info("----> Program start time: {0}".format(start_time))
    logging.info("----> Program end time: {0}".format(end_time))
    logging.info("----> Total program time taken in secs: {0}".format(end_time - start_time))
    logging.info("################################\n\n")
    logging.info("---------------- PROGRAM ENDED ----------------------")

if __name__ == "__main__":
    main()