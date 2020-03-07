# TV SHOW
TV_SHOW = 'YouNetflix_1000'

# OUTPUT DIRS
LOG_FILE = '../output/' + TV_SHOW + '_lda.log'
OUT_DIR = '../output/'
OUT_FILE = OUT_DIR + TV_SHOW

# PRE-PROCESS, LDA
PREPROCESSED_FILE = OUT_DIR + TV_SHOW + '_preprocessed_tweets_with_userid.csv'
LDA_PASSES = 20
LDA_ITERATIONS = 100
LDA_TOPICS = 10
PREPROCESS_LOG = OUT_DIR + TV_SHOW + '_preprocess.log'
LDA_LOG = OUT_DIR + TV_SHOW + '_LDA.log'

# RETWEET NETWORK
DB_NAME = 'tvshow_tweets'
STREAM_COLL = 'streaming_coll_1000'
HIST_COLL = 'historical_coll_1000'
NODE_GRAPH = './output/' + TV_SHOW + '.graphml'
RETWEET_LOG = './output/' + TV_SHOW + '_retweetNetwork.log'
DATA = 'STREAM_DATA'


# TWITTER RANK
GAMMA = 0.2
TOLERANCE = 1e-6
ITERATIONS = 100
NUM_TOPICS = 10
PRE_PROCESSED_FILE_NAME = TV_SHOW
LDA_FILE_NAME = TV_SHOW
GRAPHML_FILE = '../../notebooks/' + TV_SHOW + '.graphml'
USER_TOPIC_FILE = OUT_DIR + TV_SHOW + '_final_topic_frame.csv'
DOC_TOPIC_FILE = '../../notebooks/' + TV_SHOW + '_doc_topic_frame.csv'
TWITTER_RANK_FILE = '../../notebooks/' + TV_SHOW + '_topic_rank_frame.csv'
GRAPH_NODE_FILE = '../../notebooks/' + TV_SHOW + '_graph_nodes_with_no_isolated_nodes.csv'
NUM_OF_INFLUENTIAL_NODES = 20

# COMMUNITY DETECTION
COSINE_SIM_DIR = OUT_DIR + '{}_pickle_dumps/'.format(TV_SHOW)
