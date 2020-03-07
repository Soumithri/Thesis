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
from sklearn.preprocessing import Normalizer
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
    df = pd.read_csv(infile, names=['userid', 'tweets'], usecols=['userid'], delimiter='|')
    #df.tweets = df.tweets.apply(lambda x: literal_eval(x))
    #df['tweets'] = df.tweets.str.replace(r'\W+',' ')
# Convert the tweet_doc into tweet_tokens and remove non_alphanumeric strings in the tokens
#df['tweet_tokens'] = df['tweets'].apply(lambda x: x.split())
#%%
def plot_graph(G):
    pos = nx.spring_layout(G, k=0.3*1/np.sqrt(len(G.nodes())), iterations=20)
    nx.draw_networkx_nodes(G, pos, node_size = 50, with_labels=True)
    #nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, with_labels=True, edge_color='black', arrows=True)
    plt.rcParams['figure.figsize'] = [200, 200]
    plt.title("Retweet Network drawn from 200 random nodes", { 'fontsize': 20 })
    plt.axis('off')
    plt.rcParams["figure.figsize"] = (30,30)
    plt.show()
    
def remove_isolated_nodes(G):
    print(nx.info(G))
    print("Remove {} self loops.".format(G.number_of_selfloops()))
    G.remove_edges_from(G.selfloop_edges())
    G.remove_nodes_from(list(nx.isolates(G)))
    print(nx.info(G))
    return G

#%%
graph = nx.read_graphml(OUT_DIR + TV_SHOW + '.graphml')

#%%
#graph = remove_isolated_nodes(graph)

#%%
def add_weights(graph):
    degree_list = ['retweet_count', 'mention_count', 'reply_count', 'quote_count']
    attrs = {}
    for (node1,node2,*data) in graph.edges(data=True):
        weight = sum([value for key, value in data[0].items() if key in degree_list])
        attrs[(node1, node2)] = {'weight': weight}
    nx.set_edge_attributes(graph, attrs)
    return graph

graph = add_weights(graph)
#%%
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
    if nx.has_path(graph, nodei, nodej) and graph.has_edge(nodei, nodej):
        #print(nodei , nodej)
        return (graph.get_edge_data(nodei, nodej)['weight'] / graph.out_degree(nodei, weight='weight'))
    else:
        return 0.0

def get_Pt(DT_row_norm, k, graph, data=df):
    size = DT_row_norm.shape[0]
    trans_mat = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            if graph.has_node(str(data['userid'].iloc[i])) and graph.has_node(str(data['userid'].iloc[j])):
                trans_mat[i][j] = get_weight(str(data['userid'].iloc[i]), str(data['userid'].iloc[j]), graph) * get_sim(DT_row_norm, i, j, k)   
            else:
                trans_mat[i][j] = 0.0      
    return trans_mat


def get_TRt(gamma, trans_mat, Et, iter=1000, tolerance=1e-16):
    old_TRt = Et
    i = 0
    while i < iter:
        TRt = (gamma*np.dot(trans_mat,old_TRt)) + ((1 - gamma) * Et)
        euclidean_dis = np.linalg.norm(TRt - old_TRt)
        if euclidean_dis < tolerance: 
            print('Topic Rank vectors have converged...')
            break
        old_TRt = TRt
        i += 1
    return TRt

def get_TR(DT_row_norm, DT_col_norm, num_topics, gamma, tolerance, graph, data=df):
    for k in range(0, num_topics):
        trans_mat = get_Pt(DT_row_norm, k, graph, data)
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

#%%
doc_topic_dist = get_doc_topic_dist(OUT_DIR)
DT_row_norm = get_DT_row_norm(doc_topic_dist)
DT_col_norm = get_DT_col_norm(doc_topic_dist)
#%%
# Check the transition matrix
#%%
TR = get_TR(DT_row_norm, DT_col_norm, graph=graph, data=df, 
            num_topics=NUM_TOPICS, gamma=GAMMA, tolerance=TOLERANCE)

#%%
TR_sum = np.sum(TR, axis=1).tolist()
TR_sum = [item for sublist in TR_sum for item in sublist]

#%%
users = list()
for index in sorted(range(len(TR_sum)), key=lambda i: TR_sum[i], reverse=True):
    users.append((index, df['userid'].iloc[index], TR_sum[index]))
top_users = list()
for index in sorted(range(len(TR_sum)), key=lambda i: TR_sum[i], reverse=True)[:20]:
    top_users.append((index, df['userid'].iloc[index], TR_sum[index]))

#%%
# import random
# karate = nx.karate_club_graph().to_directed()
# for (u, v) in karate.edges():
#     karate.edges[u,v]['weight'] = random.randint(0,10)
# d = dict(karate.degree)
# options = {
#     'node_color': 'red',
#     'node_size': [v * 1000 for v in d.values()],
#     'line_color': 'black',
#     'linewidths': 5,
#     'font_size': 25,
#     'width': 0.5,
#     'label': 'Zachary Karate Club'
# }
# labels = nx.get_edge_attributes(graph,'weight')
# plt.figure(figsize=(30,30)) 
# pos = nx.circular_layout(karate)
# nx.draw_networkx_edge_labels(karate, pos=pos, with_labels=True, **options)


### PHASE 2 COMMUNITY DETECTION
#%%
steps = 3
m = graph.number_of_nodes()

#%%
PN = Normalizer(norm='l1').fit_transform(np.ones((m, m)))
P = nx.adjacency_matrix(graph, weight='weight').todense()
P = P / np.array([graph.degree(i) for i in graph.nodes])[:, None]
P = Normalizer(norm='l1').fit_transform(P)

#%%
for i in range(1, steps+1):
    PN += P**i

PN_df = pd.DataFrame(data = PN, index=graph.nodes, columns=graph.nodes)

#%%
test2 = nx.to_pandas_adjacency(graph)

#%%
from pymongo import MongoClient
client = MongoClient(host="35.225.215.122",
                     port=27017, 
                     username="root", 
                     password="twitterdb",
                    authSource="admin")
db_obj = client['stream_store']
coll = db_obj['old_tweets']

#%%
for i in coll.find_one({'user.id':2197528760}):
    print(i)


#%%
for i in coll.find({}, {'user':1}):
    print(i)
    break

#%%
from sklearn.metrics.pairwise import cosine_similarity

def cluster(seed_nodes, graph):

    # Compute adjacency matrix
    A = nx.adjacency_matrix(graph, weight='weight').todense()

    # Compute the initial transition matrix
    M = A/A.sum(axis=1)
    M[np.isnan(M)] = 0

    # Compute the initial google matrix
    P = nx.google_matrix(graph, weight='weight')
    PF = np.copy(P)
    # Compute random walk for t steps
    t = 3
    for i in range(2,t+1):
        PF += np.linalg.matrix_power(P, i)
    
    P_degree = np.diag(PF)
    P_weight = cosine_similarity(PF)

    coms = set()
    membership = {}

    # Sort the nodes array in decreasing value of weights
    sorted_P_deg = np.sort(P_degree)[::-1]
    sorted_P_deg_indices = np.argsort(P_degree)
    nodes_list = list(graph.nodes)
    sorted_nodes = [nodes_list[i] for i in sorted_P_deg_indices]

    # Take the P-degree of N/4th node as threshold
    Pt = sorted_P_deg[graph.number_of_nodes()//4]
    
    # Loop over the nodes and check if P-degree of node > Pt
    # If P-degree of a node > Pt and P-degree is not in a sub region,
    # keep the node as a seed node and map the nodes connected to as a
    # community
    # If P-degree of a node < Pt and P-degree is in a sub-region, skip

    for i, node in enumerate(sorted_nodes):
        if sorted_P_deg[i] > Pt and node not in coms:
            coms |= {node, *graph.neighbors(node)}
            membership[node] = [*graph.neighbors(node)]
    
    # Check if there are unmapped free nodes
    # If present, check if there is a connection and place it in the 
    # corresponding sub-regions
    # free_nodes = set(graph.nodes) - coms
    # if free_nodes:
    #     for node in free_nodes:
    #         if graph.neighbors(node):
    #             for neighbor in graph.neighbors(node):
    #                 graph
    
    # For every seed node, get all connected out-edges
    #members = dict(zip(seed_nodes, [[x[1] for x in graph.edges(i)] for i in seed_nodes]))
    
