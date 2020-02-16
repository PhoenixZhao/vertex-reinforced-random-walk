'''
'''
from __future__ import print_function, division

import pickle
import argparse
import os
import numpy as np
import networkx as nx
import scipy.io
from gensim.models import Word2Vec
from sklearn import metrics, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Default parameters from node2vec paper (and for DeepWalk)
default_params = {
    'log2p': 0,                     # Parameter p, p = 2**log2p
    'log2q': 0,                     # Parameter q, q = 2**log2q
    'log2d': 7,                     # Feature size, dimensions = 2**log2d
    'edge_function': "hadamard",    # Default edge function to use
    # Proportion of edges to remove nad use as positive samples
    "prop_pos": 0.5,
    "prop_neg": 0.5,                # Number of non-edges to use as negative samples
                                    #  (as a proportion of existing edges, same as prop_pos)
}

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}


def parse_mat_file(path):
    edges = []
    g = nx.Graph()
    mat = scipy.io.loadmat(path)
    nodes = mat['network'].tolil()
    subs_coo = mat['group'].tocoo()

    for start_node, end_nodes in enumerate(nodes.rows, start=0):
        for end_node in end_nodes:
            edges.append((start_node, end_node))

    g.add_edges_from(edges)
    g.name = path
    print(nx.info(g) + "\n---------------------------------------\n")

    return g, subs_coo


def model_build(model_name):
    if model_name == "deepwalk":
        from cogdl.models import deepwalk
        model = deepwalk.DeepWalk(dimension=128, walk_length=80, walk_num=40, window_size=10, worker=10, iteration=10)
    elif model_name == "line":
        from cogdl.models import line
        model = line.LINE(dimension=128, walk_length=80, walk_num=40, negative=5, batch_size=1000, alpha=0.025, order=3)
    elif model_name == "node2vec":
        from cogdl.models import node2vec
        model = node2vec.Node2vec(dimension=128, walk_length=80, walk_num=40, window_size=5, worker=10, iteration=10, p=1, q=1)
    elif model_name == "grarep":
        from cogdl.models import grarep
        model = grarep.GraRep(dimension=128, step=5)
    elif model_name == "hope":
        from cogdl.models import hope
        model = hope.HOPE(beta=0.01, dimension=128)
    elif model_name == "vrrw":
        from cogdl.models import vrrw
        model = vrrw.VRRW(dimension=args.dimension,
                          walk_length=args.walk_length,
                          walk_num=args.walk_num,
                          window_size=args.window_size,
                          worker=args.worker,
                          iteration=args.iteration,
                          alpha=args.alpha,
                          reverse=args.reverse,
                          explore=args.explore)
    else:
        from cogdl.models import drrw
        model = drrw.DRRW(dimension=args.dimension,
                          walk_length=args.walk_length,
                          walk_num=args.walk_num,
                          window_size=args.window_size,
                          worker=args.worker,
                          iteration=args.iteration,
                          alpha=args.alpha,
                          div=args.div,
                          explore=args.explore)
    print("model name:", model_name)
    return model


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run link prediction.")

    parser.add_argument('--input', nargs='?', default='graphs/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--model', nargs='?', default='drrw',
                        help='Model name')

    parser.add_argument('--regen', dest='regen', action='store_true',
                        help='Regenerate random positive/negative links')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--num-experiments', type=int, default=1,
                        help='Number of experiments to average. Default is 1.')
    parser.add_argument(
        '--unweighted',
        dest='unweighted',
        action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument(
        '--undirected',
        dest='undirected',
        action='store_false')
    parser.set_defaults(directed=False)

    #######################################################################
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument('--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 40.')
    parser.add_argument('--walk-num', type=int, default=80,
                        help='Number of walks per source. Default is 80.')
    parser.add_argument(
        '--window-size',
        type=int,
        default=10,
        help='Window size of skip-gram model. Default is 10.')
    parser.add_argument('--worker', type=int, default=10,
                        help='Number of parallel workers. Default is 10.')
    parser.add_argument('--iteration', type=int, default=10,
                        help='Number of iterations. Default is 10.')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Balance exploit-explore. Default is 0.01.')
    parser.add_argument('--div', default="kl", help='"js", "kl", "ws".')
    parser.add_argument(
        '--explore',
        default='exploration',
        help='Explore type: "cold-start", "exploration", "decay-greedy"')
    parser.add_argument('--reverse', dest='reverse', action='store_true')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(feature=True)

    return parser.parse_args()


class GraphN2V():
    def __init__(self,
                 nx_G=None, is_directed=False,
                 prop_pos=0.5, prop_neg=0.5,
                 workers=1,
                 random_seed=None):
        self.G = nx_G
        self.is_directed = is_directed
        self.prop_pos = prop_neg
        self.prop_neg = prop_pos
        self.wvecs = None
        self.workers = workers
        self._rnd = np.random.RandomState(seed=random_seed)

    def read_graph(
            self,
            input,
            enforce_connectivity=True,
            weighted=False,
            directed=False):
        '''
        Reads the input network in networkx.
        '''
        input_path = input
        if input_path.split('.')[-1] == 'edgelist':
            G = nx.read_edgelist(input_path, nodetype=int, data=(
                ('weight', float),), create_using=nx.DiGraph())
        else:
            G, _ = parse_mat_file(input_path)
            print(list(G.nodes()))
        if not weighted:
            # G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

        if not directed:
            G = G.to_undirected()

        # Take largest connected subgraph
        if enforce_connectivity and not nx.is_connected(G):
            G = max(nx.connected_component_subgraphs(G), key=len)
            print("Input graph not connected: using largest connected subgraph")

        # Remove nodes with self-edges
        # I'm not sure what these imply in the dataset
        for se in G.nodes_with_selfloops():
            G.remove_edge(se, se)

        print("Read graph, nodes: %d, edges: %d" %
              (G.number_of_nodes(), G.number_of_edges()))
        self.G = G

    def learn_embeddings(self, walks, dimensions, window_size=10, niter=5):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks,
                         size=dimensions,
                         window=window_size,
                         min_count=0,
                         sg=1,
                         workers=self.workers,
                         iter=niter)
        self.wvecs = model.wv

    def generate_pos_neg_links(self):
        """
        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.

        Modify graph by removing the postive links.
        """
        # Select n edges at random (positive samples)
        n_edges = self.G.number_of_edges()
        npos = int(self.prop_pos * n_edges)
        nneg = int(self.prop_neg * n_edges)

        if not nx.is_connected(self.G):
            raise RuntimeError("Input graph is not connected")

        non_edges = [e for e in nx.non_edges(self.G)]
        print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

        # Select m pairs of non-edges (negative samples)
        rnd_inx = self._rnd.choice(len(non_edges), nneg, replace=False)
        neg_edge_list = [non_edges[ii] for ii in rnd_inx]

        if len(neg_edge_list) < nneg:
            raise RuntimeWarning(
                "Only %d negative edges found" % (len(neg_edge_list))
            )

        print("Finding %d positive edges of %d total edges" % (npos, n_edges))

        # Find positive edges, and remove them.
        edges = list(self.G.edges())
        pos_edge_list = []
        n_count = 0
        n_ignored_count = 0
        rnd_inx = self._rnd.permutation(n_edges)
        print(rnd_inx)
        print(type(edges))
        for eii in rnd_inx:
            edge = edges[int(eii)]

            # Remove edge from graph
            data = self.G[edge[0]][edge[1]]
            self.G.remove_edge(*edge)

            # Check if graph is still connected
            # TODO: We shouldn't be using a private function for bfs
            reachable_from_v1 = nx.connected._plain_bfs(self.G, edge[0])
            if edge[1] not in reachable_from_v1:
                self.G.add_edge(*edge, **data)
                n_ignored_count += 1
            else:
                pos_edge_list.append(edge)
                print("Found: %d    " % (n_count), end="\r")
                n_count += 1

            # Exit if we've found npos nodes or we have gone through the whole
            # list
            if n_count >= npos:
                break

        if len(pos_edge_list) < npos:
            raise RuntimeWarning("Only %d positive edges found." % (n_count))

        self._pos_edge_list = pos_edge_list
        self._neg_edge_list = neg_edge_list

    def get_selected_edges(self):
        edges = self._pos_edge_list + self._neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(self._pos_edge_list)] = 1
        return edges, labels

    def train_embeddings(self):
        self.model = model_build(args.model)
        embeddings = self.model.train(self.G)

        self.wvecs = dict()
        for vid, node in enumerate(self.G.nodes()):
            self.wvecs[node] = embeddings[vid]

    def edges_to_features(self, edge_list, edge_function, dimensions):
        """
        Given a list of edge lists and a list of labels, create
        an edge feature array using binary_edge_function and
        create a label array matching the label in the list to all
        edges in the corresponding edge list

        :param edge_function:
            Function of two arguments taking the node features and returning
            an edge feature of given dimension
        :param dimension:
            Size of returned edge feature vector, if None defaults to
            node feature size.
        :param k:
            Partition number. If None use all positive & negative edges
        :return:
            feature_vec (n, dimensions), label_vec (n)
        """
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, dimensions), dtype='f')

        # Iterate over edges
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = self.wvecs[v1]
            emb2 = self.wvecs[v2]

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec


def create_train_test_graphs(args):
    """
    Create and cache train & test graphs.
    Will load from cache if exists unless --regen option is given.

    :param args:
    :return:
        Gtrain, Gtest: Train & test graphs
    """
    # Remove half the edges, and the same number of "negative" edges
    prop_pos = default_params['prop_pos']
    prop_neg = default_params['prop_neg']

    # Create random training and test graphs with different random edge
    # selections
    cached_fn = "%s.graph" % (os.path.basename(args.input))
    if os.path.exists(cached_fn) and not args.regen:
        print("Loading link prediction graphs from %s" % cached_fn)
        with open(cached_fn, 'rb') as f:
            cache_data = pickle.load(f)
        Gtrain = cache_data['g_train']
        Gtest = cache_data['g_test']

    else:
        print("Regenerating link prediction graphs")
        # Train graph embeddings on graph with random links
        Gtrain = GraphN2V(is_directed=False,
                          prop_pos=prop_pos,
                          prop_neg=prop_neg,
                          workers=args.workers)
        Gtrain.read_graph(args.input,
                          weighted=args.weighted,
                          directed=args.directed)
        Gtrain.generate_pos_neg_links()

        # Generate a different random graph for testing
        Gtest = GraphN2V(is_directed=False,
                         prop_pos=prop_pos,
                         prop_neg=prop_neg,
                         workers=args.workers)
        Gtest.read_graph(args.input,
                         weighted=args.weighted,
                         directed=args.directed)
        Gtest.generate_pos_neg_links()

        # Cache generated  graph
        cache_data = {'g_train': Gtrain, 'g_test': Gtest}
        with open(cached_fn, 'wb') as f:
            pickle.dump(cache_data, f)

    return Gtrain, Gtest


def test_edge_functions(args):
    Gtrain, Gtest = create_train_test_graphs(args)
    dimensions = 2**default_params['log2d']
    # num_walks = default_params['num_walks']
    # walk_length = default_params['walk_length']
    # window_size = default_params['window_size']

    # Train and test graphs, with different edges
    edges_train, labels_train = Gtrain.get_selected_edges()
    edges_test, labels_test = Gtest.get_selected_edges()

    # With fixed test & train graphs (these are expensive to generate)
    # we perform k iterations of the algorithm
    # TODO: It would be nice if the walks had a settable random seed
    aucs = {name: [] for name in edge_functions}
    for iter in range(args.num_experiments):
        print("Iteration %d of %d" % (iter, args.num_experiments))

        # Learn embeddings with current parameter values
        Gtrain.train_embeddings()
        Gtest.train_embeddings()

        for edge_fn_name, edge_fn in edge_functions.items():
            # Calculate edge embeddings using binary function
            edge_features_train = Gtrain.edges_to_features(
                edges_train, edge_fn, dimensions)
            edge_features_test = Gtest.edges_to_features(
                edges_test, edge_fn, dimensions)

            # Linear classifier
            scaler = StandardScaler()
            lin_clf = LogisticRegression(C=1)
            clf = pipeline.make_pipeline(scaler, lin_clf)

            # Train classifier
            clf.fit(edge_features_train, labels_train)
            auc_train = metrics.scorer.roc_auc_scorer(
                clf, edge_features_train, labels_train)

            # Test classifier
            auc_test = metrics.scorer.roc_auc_scorer(
                clf, edge_features_test, labels_test)
            aucs[edge_fn_name].append(auc_test)

    print("Edge function test performance (AUC):")
    for edge_name in aucs:
        auc_mean = np.mean(aucs[edge_name])
        auc_std = np.std(aucs[edge_name])
        print("[%s] mean: %.4g +/- %.3g" % (edge_name, auc_mean, auc_std))

    return aucs


if __name__ == "__main__":
    args = parse_args()
    test_edge_functions(args)
