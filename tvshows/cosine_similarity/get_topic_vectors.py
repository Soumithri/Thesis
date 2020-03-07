import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer

from tvshows.common.common import TV_SHOW
from tvshows.common.common import GRAPH_NODE_FILE
from tvshows.common.common import OUT_DIR
from tvshows.common.common import USER_TOPIC_FILE
from tvshows.common.common import DOC_TOPIC_FILE


def get_topic_frame(file, graph_node_file=GRAPH_NODE_FILE):
    graph = pd.read_csv(graph_node_file)
    graph = graph.rename(columns={'Unnamed: 0': 'userid'})
    topic = pd.read_csv(file)
    topic = topic.rename(columns={'Unnamed: 0': 'userid'})
    new = pd.merge(graph, topic, on = 'userid', how='left')
    dic = {'0_x': '0_y', '1_x': '1_y', '2_x': '2_y', '3_x': '3_y', '4_x': '4_y',
           '5_x': '5_y', '6_x': '6_y', '7_x': '7_y', '8_x': '8_y', '9_x': '9_y'}
    for i in dic:
        new[i] = new[i] + new[dic[i]]
    new = new.drop(columns=dic.values())
    new.set_index('userid', inplace=True)
    bool_idx = new.isnull().any(axis=1)
    new = new.mask(new.isnull(), np.random.uniform(low=0.0, high=1.0, size=new.shape))
    new[bool_idx] = Normalizer(norm='l1').fit_transform(new[bool_idx])
    return new


if __name__ == '__main__':
    topic = get_topic_frame(DOC_TOPIC_FILE, GRAPH_NODE_FILE)
    topic.to_csv(USER_TOPIC_FILE)
