import os

from sklearn.preprocessing import normalize
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import logging
from timeit import default_timer

from input_values import TV_SHOW, LOG_FILE, OUT_FILE, PREPROCESSED_FILE, LDA_PASSES, LDA_ITERATIONS, LDA_TOPICS

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(message)s')


def create_dict_corpus(doc_list, SAVE_FILE=OUT_FILE):
    '''Creates a dictionary and corpus file using a dataframe and saves the file as 'dict' file
    and 'MM corpus' file given by
    '''
    print('Directory: ' + SAVE_FILE + '.dict')
    if not os.path.exists(SAVE_FILE + '.dict'):
        print('creating dictionary and mm_corpus for the {} file..'.format(SAVE_FILE))
        dictionary = corpora.Dictionary(doc_list)
        dictionary.save(SAVE_FILE + '.dict')
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_list]
        corpora.MmCorpus.serialize(SAVE_FILE + '.mm', doc_term_matrix)
        mm_corpus = corpora.MmCorpus(SAVE_FILE + '.mm')
    else:
        print('Dictionary and mm_corpus for the file:{} already exists.. loading..'.format(SAVE_FILE))
        dictionary = corpora.Dictionary.load(SAVE_FILE + '.dict')
        mm_corpus = corpora.MmCorpus(SAVE_FILE + '.mm')
    return (dictionary, mm_corpus)


def get_lda_model(corpus, dictionary, num_topics, SAVE_FILE=OUT_FILE, passes=20, iterations=100):
    if not os.path.exists(SAVE_FILE + '.lda'):
        print('creating lda model for the {} file..'.format(SAVE_FILE))
        print('num_topics: {}'.format(num_topics))
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                                 passes=passes, iterations=iterations, chunksize=2500)
        lda_model.save(SAVE_FILE + '.lda')
    else:
        print('LDA model for the file:{} already exists.. loading..'.format(SAVE_FILE))
        lda_model = LdaMulticore.load(SAVE_FILE + '.lda')
    return lda_model


if __name__ == '__main__':
    # Start the timer
    start_time = default_timer()

    # Load the data into the dataframe
    with open(PREPROCESSED_FILE, 'r') as infile:
        df = pd.read_csv(infile, names=['userid', 'tweets'], delimiter='|')
        df['tweets'] = df.tweets.str.replace(r'\W+', ' ')

    # Create tokens
    df['tweet_tokens'] = df['tweets'].apply(lambda x: x.split())

    # Create the dictionary and mmcorpus
    (dictionary, corpus) = create_dict_corpus(list(df['tweet_tokens']), SAVE_FILE=OUT_FILE)
    print('Num_topics: {}'.format(LDA_TOPICS))
    print('Computing LDA model...')
    lda_model = get_lda_model(corpus=corpus, dictionary=dictionary, SAVE_FILE=OUT_FILE,
                              num_topics=LDA_TOPICS, passes=LDA_PASSES,
                              iterations=LDA_ITERATIONS)
    print('Finished computing LDA model...')
    end_time = default_timer()
    total_time = (end_time - start_time) / 60
    logging.info("----> Total program time taken: {0} mins".format(total_time))
