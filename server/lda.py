import logging
import os
from collections import namedtuple

import pandas as pd
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from timeit import default_timer

# Output Directories to save
OUTPUT_DIR = '../models'
TV_SHOW = 'NCISNOLA'

def create_Directories(path):
  ''' Checks for the directory. If not present creates the directory'''
  try: 
      os.makedirs(path)
  except OSError:
      if not os.path.isdir(path):
          raise

def get_data():
  with open('../output/'+TV_SHOW+'_preprocessed_tweets_with_userid.csv', 'r') as infile:
    df = pd.read_csv(infile, names=['userid', 'tweets'], delimiter='|')
    df['tweets'] = df.tweets.str.replace(r'\W+',' ')
    df['tweet_tokens'] = df['tweets'].apply(lambda x: x.split())
  return df

def create_dict_corpus(doc_list, SAVE_DIR):
  '''Creates a dictionary and corpus file using a dataframe and saves the file as 'dict' file 
  and 'MM corpus' file given by TV_SHOW
  '''
  if not os.path.exists(SAVE_DIR + '/' + TV_SHOW +'.dict'):
      dictionary = corpora.Dictionary(doc_list)
      dictionary.save(SAVE_DIR + '/' + TV_SHOW +'.dict')
      doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_list]
      corpora.MmCorpus.serialize(SAVE_DIR + '/' + TV_SHOW + '.mm', doc_term_matrix)
      mm_corpus = corpora.MmCorpus(SAVE_DIR + '/' + TV_SHOW + '.mm')
  else:
      dictionary = corpora.Dictionary.load(SAVE_DIR + '/' + TV_SHOW + '.dict')
      mm_corpus = corpora.MmCorpus(SAVE_DIR + '/' + TV_SHOW + '.mm')
  return (dictionary, mm_corpus)

def run_lda(corpus, dictionary, texts, num_topics = 10, passes=20, iterations=100):
  eval_frame = pd.DataFrame(columns=['Num_Topics','Log_Perplexity_P_{0}_I_{1}'.format(passes, iterations), 
                                     'Topic_Coherence(u_mass)_P_{0}_I_{1}'.format(passes, iterations),
                                     'Topic_Coherence(c_uci)_P_{0}_I_{1}'.format(passes, iterations),
                                     'Topic_Coherence(c_v)_P_{0}_I_{1}'.format(passes, iterations),
                                     'Topic_Coherence(c_npmi)_P_{0}_I_{1}'.format(passes, iterations)])
  logging.debug('******* RUNNING LDA *************')
  lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, 
                           iterations=iterations, chunksize=2500)
  coh_model_umass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
  coh_model_uci = CoherenceModel(model=lda_model, texts=texts, coherence='c_uci')
  coh_model_ucv = CoherenceModel(model=lda_model, texts=texts, coherence='c_v')
  coh_model_npmi = CoherenceModel(model=lda_model, texts=texts, coherence='c_npmi')
  eval_frame.loc[len(eval_frame)] = [num_topics, lda_model.log_perplexity(corpus), coh_model_umass.get_coherence(), 
                                      coh_model_uci.get_coherence(), coh_model_ucv.get_coherence(),coh_model_npmi.get_coherence()]
  model = namedtuple('model',['lda_model', 'eval_frame'])
  return model(lda_model, eval_frame)
   

def main(dictionary, corpus, texts, num_topics, passes, iterations, SAVE_DIR):
  logging.debug('Running LDA and Topic Coherence ...\n')
  model = run_lda(corpus=corpus, dictionary=dictionary, texts=texts,  num_topics=num_topics, 
                   passes=passes, iterations=iterations)
  model.lda_model.save(SAVE_DIR+'/'+ TV_SHOW +'.lda', separately=False)
  df_p20_i100 = model.eval_frame
  df_p20_i100.to_csv(SAVE_DIR+'/'+TV_SHOW+'_values.csv', index=False)

if __name__=='__main__':   
  SAVE_DIR = OUTPUT_DIR+'/'+TV_SHOW
  create_Directories(SAVE_DIR)
  log_file = SAVE_DIR+'/'+TV_SHOW+'.log'
  logging.basicConfig(filename=log_file, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  logging.debug('************* STARTING PROGRAM **************')
  df = get_data() 
  start_time = default_timer()
  (dictionary, corpus) = create_dict_corpus(list(df['tweet_tokens']), SAVE_DIR)
  model = main(dictionary=dictionary, corpus=corpus, texts=list(df['tweet_tokens']), num_topics=10, passes=20, iterations=100, 
                SAVE_DIR=SAVE_DIR)
  end_time = default_timer()
  logging.debug("----> Total program time taken in secs: {0}".format(end_time - start_time))