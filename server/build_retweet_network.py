import networkx as nx
from matplotlib import pyplot as plt
import json
import sys
from pprint import pprint
from collections import namedtuple

def build_retweet_network(internal_only=True):
    all_users = set()
    result_set_users = set()
    Edge = namedtuple("Edge", ["tweeter", "retweeted"])
    retweet_dict = {}

    print("Getting user data to build retweet network...")

    with open('./output/retweets_data2.json','r') as f:
        for data in f:
            retweet = json.loads(data)
            #pprint(retweet)
            tweet_user = retweet['user']['screen_name']
            retweet_user = retweet['retweeted_status']['user']['screen_name']  
            if not retweet_user:
                print("Warning: No retweeted user info for tweet (text: {0})".format(retweet['text']))
                continue
            #print(tweet_user, retweet_user)
            all_users.add(tweet_user)
            all_users.add(retweet_user)
            result_set_users.add(tweet_user)

            e = Edge(tweeter=tweet_user, retweeted=retweet_user)
            if e in retweet_dict:
                retweet_dict[e] += 1
            else:
                retweet_dict[e] = 1

    print( "Summary:\n")
    print( "{0} total users tweeted or retweeted".format(len(all_users)))
    print( "{0} total users tweeted".format(len(result_set_users)))
    print( "{0} external users (users that were retweeted but did not tweet in the data set)".format(
            len(all_users.difference(result_set_users))))
    print( "{0} directed edges between users".format(len(retweet_dict)))
    print( "{0} total retweets between all users (including retweeting from same users more than once)\n".format(
            sum(retweet_dict.values())))

    print( "Building DiGraph...")
    # Create digraph
    DG = nx.DiGraph()

    if internal_only:
        # Add all result_set users only to the graph, with type and color properties
        DG.add_nodes_from(list(result_set_users), node_type="internal", color="#2A2AD1")

        # Add all edges where both tweeter and retweeted are in the user list
        for edge in retweet_dict.items():
            if edge[0].tweeter in result_set_users and edge[0].retweeted in result_set_users:
                DG.add_edge(edge[0].tweeter, edge[0].retweeted, weight=edge[1])

    else:
        # Add all users as nodes, with type property internal (tweeted in result set) or external (just a retweeted user)
        DG.add_nodes_from(list(result_set_users), node_type="internal", color="#2A2AD1")
        DG.add_nodes_from(list(all_users.difference(result_set_users)), node_type="external", color="#CCCCCC")

        # Add all edges, with weight property equal to number of directional retweets between two users (no edge for 0)
        DG.add_weighted_edges_from([ (e[0].tweeter, e[0].retweeted, e[1]) for e in retweet_dict.items() ])

    # Return graph/network!
    return DG

def display_retweet_network(network, outfile=None, show=False):
    """
    Take a DiGraph (retweet network?) and display+/save it to file.
    Nodes must have a 'color' property, represented literally and indicating their type
    Edges must have a 'weight' property, represented as edge width
    """

    # Create a color list corresponding to nodes.
    node_colors = [ n[1]["color"] for n in network.nodes(data=True) ]

    # Get edge weights from graph
    edge_weights = [ e[2]["weight"] for e in network.edges(data=True) ]

    # Build up graph figure
    #pos = nx.random_layout(network)
    pos = nx.spring_layout(network)
    nx.draw_networkx_edges(network, pos, alpha=0.3 , width=edge_weights, edge_color='m')
    nx.draw_networkx_nodes(network, pos, node_size=400, node_color=node_colors, alpha=0.4)
    #nx.draw_networkx_labels(network, pos, fontsize=6)

    plt.title("Retweet Network", { 'fontsize': 12 })
    plt.axis('off')

    if outfile:
        print("Saving network to file: {0}".format(outfile))
        plt.savefig(outfile)

    if show:
        print("Displaying graph. Close graph window to resume python execution")
        plt.show()
                
def main():

    rn = build_retweet_network(internal_only=True)
    display_retweet_network(rn, './output/retweet_network.png', show=True)

if __name__=='__main__':
    main()    