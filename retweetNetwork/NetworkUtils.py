import networkx as nx
import random
import logging


def sample(g, percent):
    """ Create a sample network from g.

    :param g:           Original network
    :param percent:     Percentage of nodes in the sample network
    """
    assert isinstance(g, nx.Graph), "Network must be instance of graph"
    assert 0 < percent < 1, "Percentage must between 0 and 1"

    nodes = g.nodes()
    nodes_sample = random.sample(nodes, int(percent * len(nodes)))
    sg = g.subgraph(nodes_sample)
    return sg


def graph_info(g):
    """ Show network nodes and edges

    :param g:   Network
    """
    print("Graph nodes: {}, edges: {}".format(g.number_of_nodes(), g.number_of_edges()))
    logging.info("Graph nodes: {}, edges: {}".format(g.number_of_nodes(), g.number_of_edges()))


def get_graphml(input_file):
    """ Read network from graphml format

    :param input_file:  Graphml file
    :return:            Graph
    """
    g = nx.read_graphml(input_file)
    graph_info(g)
    print("Remove {} self loops.".format(g.number_of_selfloops()))
    g.remove_edges_from(g.selfloop_edges())
    # mapping the node labels from string to integer
    g = nx.relabel_nodes(g, lambda x:int(x))
    graph_info(g)
    return g
