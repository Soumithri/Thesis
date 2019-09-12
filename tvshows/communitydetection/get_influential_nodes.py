# Jupyter Notebook VS code for Twitter Rank

#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'tvshows/communitydetection/'))
	print(os.getcwd())
except:
    print(os.getcwd())

from IPython import get_ipython

#%%
import numpy as np
from sklearn.preprocessing import normalize
from gensim import corpora, models, matutils
from gensim.models.ldamulticore import LdaMulticore
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import issparse
from ast import literal_eval
from pymongo import MongoClient

#%%
from input_values import TV_SHOW, OUT_DIR, GAMMA, TOLERANCE, ITERATIONS, NUM_TOPICS

#%%
# make matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')

#%%
# load the data frame
with open(OUT_DIR + TV_SHOW + '_preprocessed_tweets_with_userid.csv', 'r') as infile:
    df = pd.read_csv(infile, names=['userid', 'tweets'], delimiter='|')
    df.tweets = df.tweets.apply(lambda x: literal_eval(x))
    #df['tweets'] = df.tweets.str.replace(r'\W+',' ')
# Convert the tweet_doc into tweet_tokens and remove non_alphanumeric strings in the tokens
#df['tweet_tokens'] = df['tweets'].apply(lambda x: x.split())
#%%
def plot_graph(G):
    pos = nx.spring_layout(G, k=0.3*1/np.sqrt(len(G.nodes())), iterations=20)
    nx.draw_networkx_nodes(G, pos, node_size = 50)
    #nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True)
    plt.rcParams['figure.figsize'] = [200, 200]
    plt.title("Retweet Network drawn from 200 random nodes", { 'fontsize': 20 })
    plt.axis('off')
    plt.rcParams["figure.figsize"] = (30,30)
    plt.show()
    
def plot_sub_graph(G):
    pass
    
def remove_isolated_nodes(G):
    print(nx.info(G))
    isolated_nodes = list(nx.isolates(G))
    print('\nIsolated nodes: {}\n'.format(len(isolated_nodes)))
    print('removing isolated nodes...\n')
    G.remove_nodes_from(isolated_nodes)
    print(nx.info(G))
    return G

def get_all_degrees(G):
    return sorted(G.degree, key=lambda x: x[1], reverse=True)

def get_all_in_degrees(G):
    return sorted(G.in_degree, key=lambda x: x[1], reverse=True)

def get_degree(node, G):
    return G.degree[node]

def get_in_degree(node, G):
    return G.in_degree[node]

#%%
graph = nx.read_graphml(OUT_DIR + TV_SHOW + '.graphml')
graph = remove_isolated_nodes(graph)
#%%
# Create a sub graph with 500 nodes for plotting
np.random.seed(seed=2)
sub_G = graph.subgraph(np.random.choice(graph.nodes, 1000).tolist())
print(nx.info(sub_G))
plot_graph(sub_G)

#%%
# This functions takes the LDA topic model and returns document topic vectors
def _extract_data(topic_model, corpus, dictionary, doc_topic_dists=None):

    if not matutils.ismatrix(corpus):
        corpus_csc = matutils.corpus2csc(corpus, num_terms=len(dictionary))
    else:
        corpus_csc = corpus
        # Need corpus to be a streaming gensim list corpus for len and inference functions below:
        corpus = matutils.Sparse2Corpus(corpus_csc)

    beta = 0.01
    fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)
    term_freqs = corpus_csc.sum(axis=1).A.ravel()[fnames_argsort]
    term_freqs[term_freqs == 0] = beta
    doc_lengths = corpus_csc.sum(axis=0).A.ravel()

    assert term_freqs.shape[0] == len(dictionary), 'Term frequencies and dictionary have different shape {} != {}'.format(
        term_freqs.shape[0], len(dictionary))
    assert doc_lengths.shape[0] == len(corpus), 'Document lengths and corpus have different sizes {} != {}'.format(
        doc_lengths.shape[0], len(corpus))

    if hasattr(topic_model, 'lda_alpha'):
        num_topics = len(topic_model.lda_alpha)
    else:
        num_topics = topic_model.num_topics

    if doc_topic_dists is None:
        # If its an HDP model.
        if hasattr(topic_model, 'lda_beta'):
            gamma = topic_model.inference(corpus)
        else:
            gamma, _ = topic_model.inference(corpus)
        doc_topic_dists = gamma / gamma.sum(axis=1)[:, None]
    else:
        if isinstance(doc_topic_dists, list):
            doc_topic_dists = matutils.corpus2dense(doc_topic_dists, num_topics).T
        elif issparse(doc_topic_dists):
            doc_topic_dists = doc_topic_dists.T.todense()
        doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)

    assert doc_topic_dists.shape[1] == num_topics, 'Document topics and number of topics do not match {} != {}'.format(
        doc_topic_dists.shape[1], num_topics)

    # get the topic-term distribution straight from gensim without
    # iterating over tuples
    if hasattr(topic_model, 'lda_beta'):
        topic = topic_model.lda_beta
    else:
        topic = topic_model.state.get_lambda()
    topic = topic / topic.sum(axis=1)[:, None]
    topic_term_dists = topic[:, fnames_argsort]

    assert topic_term_dists.shape[0] == doc_topic_dists.shape[1]

    return doc_topic_dists

def get_doc_topic_dist(OUT_DIR=OUT_DIR):
    lda_dict = corpora.Dictionary.load(OUT_DIR + TV_SHOW + '.dict') 
    lda_corpus = corpora.MmCorpus(OUT_DIR + TV_SHOW + '.mm')
    lda = LdaMulticore.load(OUT_DIR + TV_SHOW + '.lda')
    return _extract_data(topic_model=lda, dictionary=lda_dict, corpus=lda_corpus)
#%%
def get_DT_row_norm(doc_topic_dist):
    DT_row_norm = np.asmatrix(normalize(doc_topic_dist, axis=1, norm='l1'))
    return DT_row_norm

def get_DT_col_norm(doc_topic_dist):
    DT_col_norm = np.asmatrix(normalize(doc_topic_dist, axis=0, norm='l1'))
    return DT_col_norm

def get_sim(DT_row_norm, i, j, k):
    sim = 1 - abs(DT_row_norm.item((i,k))-DT_row_norm.item(j,k))
    return sim    

def get_weight(nodei, nodej, graph):
    ''' Adds weights to the Transition matrix by accepting two nodes: node1, nodej.
    weight is computed as follows:
    
        weight = (sum of weighted in-degrees of nodej)/(sum of weighted degrees of node1)
        Returns 0.0 if both numerator and denominator of the above expression is 0
    '''
    degree_list = ['retweet_count', 'mention_count', 'reply_count', 'quote_count']
    nodej_in_degree = sum([graph.in_degree(nodej, weight=value) for value in degree_list])
    nodei_degree = sum([graph.degree(nodei, weight=value) for value in degree_list])
    if nodei_degree==0 or nodej_in_degree==0:
        return 0.0
    else: 
        return nodej_in_degree/nodei_degree

def get_Pt(DT_row_norm, k, data=df):
    size = DT_row_norm.shape[0]
    trans_mat = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            if graph.has_node(str(data['userid'].iloc[i])) and graph.has_node(str(data['userid'].iloc[j])):
                #trans_mat[i][j] = get_weight(str(data['userid'].iloc[i]), str(data['userid'].iloc[j]), graph) * get_sim(DT_row_norm, i, j, k)
                trans_mat[i][j] = get_sim(DT_row_norm, i, j, k)    
            else:
                trans_mat[i][j] = 0.0      
    return trans_mat

def get_DT_column_norm_to_list(doc_topic_dist):
    temp = np.array(normalize(doc_topic_dist, axis=0, norm='l1'))
    mat = temp.reshape(temp.shape).tolist()
    return mat

def get_TRt(gamma, trans_mat, Et, iter=10, tolerance=1e-16):
    old_TRt = Et
    i = 0
    while i < iter:
        TRt = (gamma*np.dot(trans_mat,old_TRt)) + ((1 - gamma) * Et)
        euclidean_dis = np.linalg.norm(TRt - old_TRt)
        if euclidean_dis < tolerance: break
        old_TRt = TRt
        i += 1
    return TRt

def get_TR(DT_row_norm, DT_col_norm, num_topics, gamma, tolerance, data=df):
    for k in range(0, num_topics):
        trans_mat = get_Pt(DT_row_norm, k, data)
        Et = DT_col_norm[:,k]
        if k==0: TR = get_TRt(gamma, trans_mat, Et)
        else: TR = np.concatenate((TR, get_TRt(gamma, trans_mat, Et)), axis=1)
    return TR

def get_TR_sum(TR, samples, num_topics):
    TR_sum = [0 for i in range(0, samples)]
    for i in range(0, num_topics):
        for j in range(0, samples):
            TR_sum[j] += TR[i][j]
    TR_sum.sort()
    return TR_sum

#%% [Markdown]
# Create the document topic distribution matrix
#%%
doc_topic_dist = get_doc_topic_dist(OUT_DIR)
DT_row_norm = get_DT_row_norm(doc_topic_dist)
DT_col_norm = get_DT_col_norm(doc_topic_dist)
#%%
# Check the transition matrix
trans_mat = get_Pt(DT_row_norm, 0, data=df)
#%%

TR = get_TR(DT_row_norm, DT_col_norm, data=df, num_topics=NUM_TOPICS, gamma=GAMMA, tolerance=TOLERANCE)
TR_sum = np.sum(TR, axis=1).tolist()
TR_sum = [item for sublist in TR_sum for item in sublist]

#%%
# users = list()
# for index in sorted(range(len(TR_sum)), key=lambda i: TR_sum[i], reverse=True):
#     users.append((index, df['userid'].iloc[index]))
# top_users = list()
# for index in sorted(range(len(TR_sum)), key=lambda i: TR_sum[i], reverse=True)[:NUM_TOPICS]:
#     top_users.append((index, df['userid'].iloc[index]))

#%%
G = nx.karate_club_graph()

#%%
d = dict(G.degree)
options = {
    'node_color': 'red',
    'node_size': [v * 1000 for v in d.values()],
    'line_color': 'black',
    'linewidths': 5,
    'font_size': 25,
    'width': 0.5,
    'label': 'Zachary Karate Club'
}
pos = nx.circular_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=True, **options)
#%%
A = nx.adjacency_matrix(G).todense()

#%%
num_topics = 3
DT_mat = np.random.uniform(0.0, 1.0, (len(G.nodes),num_topics))

#%%
def sim(DT_row_norm, i, j, k):
    sim = 1 - abs(DT_row_norm.item((i,k))-DT_row_norm.item(j,k))
    return sim 

def weight(G, i, j):
    return G.degree(j)/sum([G.degree(x) for x in G.neighbors(i)])

def get_trans_mat(DT_row_norm, k, G):
    A = nx.adjacency_matrix(G).todense()
    (m, n) = np.shape(A)
    for i in range(0, m):
        for j in range(0, n):
              A[i][j] *= weight(G, i, j) * sim(DT_row_norm, i, j, k)
    return A

def get_con(gamma, trans_mat, Et):
    tolerance = 1.0e-6
    iter = 100
    old_TRt = Et
    i = 0
    n = np.shape(trans_mat)[0]
    ro, r = np.zeros(n), np.ones(n)
    while i < iter:
        TRt = (gamma*np.dot(trans_mat,old_TRt)) + ((1 - gamma) * Et)
        euclidean_dis = np.linalg.norm(TRt - old_TRt)
        if euclidean_dis < tolerance: break
        old_TRt = TRt
        i += 1
    return TRt

def topic_rank(DT_mat, G):
    gamma = 0.2
    tolerance = 1.0e-6
    (m, _) = np.shape(DT_mat) 
    DT_row_norm = np.asmatrix(normalize(DT_mat, axis=1, norm='l1'))
    DT_col_norm = np.asmatrix(normalize(DT_mat, axis=0, norm='l1'))
    for k in range(0, m):
        trans_mat = get_trans_mat(DT_row_norm, k, G)
        Et = DT_col_norm[:,k]
        if k==0: 
            TR = get_con(gamma, trans_mat, Et)
        else: 
            TR = np.concatenate((TR, get_TRt(gamma, trans_mat, Et)), axis=1)
    return TR

