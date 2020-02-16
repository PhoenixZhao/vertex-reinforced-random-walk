import numpy as np
from gensim.models import Word2Vec
import random
from . import BaseModel, register_model
import copy
from sklearn.preprocessing import normalize


@register_model("vrrw_list")
class VRRW(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--walk-length', type=int, default=80,
                            help='Length of walk per source. Default is 80.')
        parser.add_argument('--walk-num', type=int, default=40,
                            help='Number of walks per source. Default is 40.')
        parser.add_argument(
            '--window-size',
            type=int,
            default=5,
            help='Window size of skip-gram model. Default is 5.')
        parser.add_argument('--worker', type=int, default=10,
                            help='Number of parallel workers. Default is 10.')
        parser.add_argument('--iteration', type=int, default=10,
                            help='Number of iterations. Default is 10.')
        parser.add_argument('--alpha', type=float, default=0,
                            help='Balance exploit-explore. Default is 0.')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.walk_length,
            args.walk_num,
            args.window_size,
            args.worker,
            args.iteration,
            args.alpha)

    def __init__(
            self,
            dimension,
            walk_length,
            walk_num,
            window_size,
            worker,
            iteration,
            alpha):
        super(VRRW, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration
        self.alpha = alpha

    def train(self, G):
        self.G = G
        walks = self._simulate_walks(self.walk_length, self.walk_num)
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(
            walks,
            size=self.dimension,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.worker,
            iter=self.iteration)
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([model[str(id2node[i])]
                                 for i in range(len(id2node))])
        return embeddings

    def _get_transition_probability(self, cur_nodes, memory):
        # Simulate a random walk starting from start node.
        all_nodes = np.array(self.G.nodes())
        # A = np.array(nodes)
        cur_memory = memory[np.where(np.isin(all_nodes, cur_nodes))]
        # print("cur_memory:", cur_memory)
        probs = cur_memory/sum(cur_memory)
        return probs

    def _vertex_reinforced_walk(self, start_node, walk_length, memory):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        memory[start_node] += 1
        G = self.G
        nodes = list(G.nodes())
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            probabilities = self._get_transition_probability(cur_nbrs, memory)
            _next = np.random.choice(cur_nbrs, p=probabilities)
            walk.append(_next)
            _next_ind = nodes.index(_next)
            memory[_next_ind] += 1
        return walk

    def _simulate_walks(self, walk_length, num_walks):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('node number:', len(nodes))

        count = 0
        for i in range(len(nodes)):
            if i in nodes:
                count += 1
        if count == len(nodes):
            print("Can use the index!")
        else:
            print("Cannot use the index!")
        # initialize occupation vector
        initial_vector = np.ones((len(nodes),), dtype=int)
        """
        initial_vector = dict()
        for node in G.nodes():
            initial_vector[node] = 1
        """

        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                occupation_vector = copy.copy(initial_vector)  # initialization
                walks.append(
                    self._vertex_reinforced_walk(node, walk_length, occupation_vector))
        return walks
