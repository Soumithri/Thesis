# -*- coding: utf-8 -*-
"""LDA1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QN-7MKF-tIuAAzubXQOnRc5NagzS75cv

# <tb>GENSIM LDA Visualization (local Server)</tb>

## 1. Import the required modules
"""

import logging
import re
import json
import os
from collections import namedtuple

import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import train_test_split
from timeit import default_timer
# import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
# %matplotlib inline

# Output Directories to save
OUTPUT_DIR = './models'

def create_Directories(path):
  ''' Checks for the directory. If not present creates the directory'''
  try: 
      os.makedirs(path)
  except OSError:
      if not os.path.isdir(path):
          raise

print('Current path of working directory: {}'.format(os.getcwd()))

"""## 2. Load the data into Pandas Data Frame

> ###  <font color=green>2.1. Fetch the file and load it in pandas data frame</font>
"""

def get_data():

  with open('././output/preprocessed_tweets_with_userid.csv', 'r') as infile:
    df = pd.read_csv(infile, names=['userid', 'tweets'], delimiter='|')
    df['tweets'] = df.tweets.str.replace(r'\W+',' ')
    


  """> ### <font color=green>2.2 Convert the tweet_doc into tweet_tokens</font>"""

  # Convert the tweet_doc into tweet_tokens and remove non_alphanumeric strings in the tokens

  df['tweet_doc'] = df['tweets'].apply(lambda x: x.split())
  #logging.info('Length of total dataset: {}'.format(len(df)))


  """> ### <font color=green>2.3. Split the data set into 4 sets - 20%, 40%, 60%, 80%</font>"""

  df_80, df_20 = train_test_split(df, test_size=0.2)
  df_60, df_40 = train_test_split(df, test_size=0.4)

  return (df_20, df_40, df_60, df_80)

#logging.debug('Length of df_20, df_40, df_60, df_80: {0},{1},{2},{3}'.format(len(df_20), len(df_40), len(df_60), len(df_80)))

"""> ### <font color=green>2.4. Create and save the dictionary and Mm_corpus (term-document frequency) model</font>"""

def create_dict_corpus(doc_list, fname, OUTPUT_DIR=OUTPUT_DIR):
  '''Creates a dictionary and corpus file using a dataframe and saves the file as 'dict' file 
  and 'MM corpus' file given by fname
  '''
  if not os.path.exists(OUTPUT_DIR + '/'+ fname + '.dict'):
      dictionary = corpora.Dictionary(doc_list)
      dictionary.save(OUTPUT_DIR + '/' + fname + '.dict')
      doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_list]
      corpora.MmCorpus.serialize(OUTPUT_DIR + '/' + fname + '.mm', doc_term_matrix)
      mm_corpus = corpora.MmCorpus(OUTPUT_DIR + '/' + fname + '.mm')
  else:
      dictionary = corpora.Dictionary.load(OUTPUT_DIR + '/' + fname + '.dict')
      mm_corpus = corpora.MmCorpus(OUTPUT_DIR + '/' + fname + '.mm')
  return (dictionary, mm_corpus)

"""## 3. Apply LDA - 20% data set randomly selected

> ### <font color=green>3.1. Define LDA Multicore on the corpus</font>
"""

def run_lda(corpus, dictionary, start_topic=10, end_topic=100, step_size_of_topic=10, passes=1, iterations=50):
  lda_model = dict()
  coh_model = dict()
  eval_frame = pd.DataFrame(columns=['Number of Topics','Log_Perplexity_Pass_{0}_Iter_{1}'.format(passes, iterations), 
                                     'Topic_Coherence_Pass_{0}_Iter_{1}'.format(passes, iterations)])
  logging.debug('******* RUNNING LDA *************')
  for i in range(start_topic, end_topic+1, step_size_of_topic):
    lda_model[i] = LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=passes, iterations=iterations, chunksize=2500)
    coh_model[i] = CoherenceModel(model=lda_model[i], corpus=corpus, dictionary=dictionary, coherence='u_mass')
    eval_frame.loc[len(eval_frame)] = [i, lda_model[i].log_perplexity(corpus), coh_model[i].get_coherence()]
  models = namedtuple('models',['lda_models', 'coh_models', 'eval_frame'])
  return models(lda_model, coh_model, eval_frame)
   
def save_model(*lda_model, DIR):
  #create_Directories(DIR)
  print(*lda_model)
  for num_topics, model in dict(*lda_model).items():
    print(num_topics, model)
    model.save(DIR+'/'+str(num_topics)+'.lda', separately=False)

"""> ### <font color=green>3.2. Run the LDA  & topic coherence model</font>"""

def main(dictionary, corpus):
  print('Enter the LDA parameters...\n')
  start_topic, end_topic, step_size_of_topic, passes, iterations = map(int, input('Start Topic, End_topic, step_size, passes, iterations : ').split())
  prompt = str(input('Do you wish to run LDA (y/n)? : ')).lower()
  if prompt=='y':
    logging.debug('Running LDA and Topic Coherence ...\n')
    models = run_lda(corpus=corpus, dictionary=dictionary, start_topic=start_topic, end_topic=end_topic, 
                     step_size_of_topic=step_size_of_topic, passes=passes, iterations=iterations) 
  return models

"""> ### <font color=green>3.3. Define the plot function</font>"""

def plot(df, x1, x2, y1, y2, title1, title2, save=False, figsize=(20,5), OUTPUT_DIR=OUTPUT_DIR):
  fig, axes = plt.subplots(nrows=1, ncols=2)
  ax1 = df.plot(ax=axes[0], x=x1, y=y1, color='blue',
          figsize=figsize, title='Number of Topics(K) Vs Log Perplexity')
  #ax1.set(xlabel=xlabel1, ylabel=ylabel1)
  
  ax2 = df.plot(ax=axes[1], x=x2, y=y2, color='red',
          figsize=figsize, title='Number of Topics(K) Vs Topic Coherence')
  #ax2.set(xlabel=xlabel2, ylabel=ylabel2)
  if save==True:
    create_Directories(OUTPUT_DIR)
    plt.savefig(OUTPUT_DIR+'/TopicsVsPerplexity&Coherence.png')

"""## 4. Run LDA and Plot Graphs

> ### <font color=green>4.1. 20% CORPUS</font>
"""

if __name__=='__main__':

  SAVE_DIR = OUTPUT_DIR+'/df_20_p30_iter_50'
  log_file = SAVE_DIR + '/lda_gensim.log'
  create_Directories(SAVE_DIR)
  logging.basicConfig(filename=log_file, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.debug('************* STARTING PROGRAM **************')
  (df_20, df_40, df_60, df_80) = get_data()
  start_time = default_timer()
  (dictionary, corpus) = create_dict_corpus(list(df_20['tweet_doc']), 'data_20', OUTPUT_DIR=SAVE_DIR)
  models = main(dictionary, corpus)
  df_20_p_30_i_50 = models.eval_frame
  #print(df_20_p_20_i_50)
  df_20_p_30_i_50.to_csv(SAVE_DIR+'/values.csv', index=False)
  plot(df_20_p_30_i_50, x1='Number of Topics', y1='Log_Perplexity_Pass_30_Iter_50', x2='Number of Topics', y2='Topic_Coherence_Pass_30_Iter_50', 
      title1='Number of Topics(K) Vs Log Perplexity', title2='Number of Topics(K) Vs Topic Coherence', OUTPUT_DIR=SAVE_DIR, save=True)
  save_model(models.lda_models, DIR=SAVE_DIR)
  end_time = default_timer()
  logging.debug("----> Total program time taken in secs: {0}".format(end_time - start_time))
  # num_topics = int(input('Pick a value of number of topics - [10,20,30,40,50,60,70,80,90,100]: '))
  # pyLDAvis.gensim.prepare(models.lda_models[num_topics], corpus, dictionary)