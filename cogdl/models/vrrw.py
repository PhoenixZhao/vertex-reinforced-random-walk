import numpy as np
from gensim.models import Word2Vec
from . import BaseModel, register_model
import copy
import multiprocessing as mp
from itertools import repeat
import random
from scipy.special import softmax


@register_model("vrrw")
class VRRW(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--walk-length', type=int, default=40,
                            help='Length of walk per source. Default is 40.')
        parser.add_argument('--walk-num', type=int, default=80,
                            help='Number of walks per source. Default is 80.')
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
        parser.add_argument('--explore', default='cold-start', help='Explore type: "cold-start", "exploration"')
        parser.add_argument('--reverse', dest='reverse', action='store_true')
        parser.add_argument('--no-reverse', dest='reverse', action='store_false')
        parser.set_defaults(reverse=False)
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
            args.alpha,
            args.reverse,
            args.explore)

    def __init__(
            self,
            dimension,
            walk_length,
            walk_num,
            window_size,
            worker,
            iteration,
            alpha,
            reverse,
            explore):
        super(VRRW, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration
        self.alpha = alpha
        self.reverse = reverse
        self.explore = explore

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

    def _get_transition_probability(self, nodes, memory):
        # Simulate a random walk starting from start node.
        s_n = list()
        for node in nodes:
            s_n.append(memory[node])
        probs = [float(i) / sum(s_n) for i in s_n]
        return probs

    def _get_reverse_transition_probability(self, nodes, memory):
        # Simulate a random walk starting from start node.
        s_n = list()
        for node in nodes:
            s_n.append(0.9**memory[node])
        probs = [float(i) / sum(s_n) for i in s_n]
        return probs

    def _get_all_transition_probability(self, nodes, memory):
        # Simulate a random walk starting from start node.
        s_n = list()
        for node in nodes:
            s_n.append(memory[node] + self.alpha * 0.9**memory[node])
        # probs = [float(i) / sum(s_n) for i in s_n]
        probs = softmax(s_n)
        return probs

    def _vertex_reinforced_walk(self, start_node, walk_length, memory):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        memory[start_node] += 1

        if self.explore == "cold-start":
            # print("Epsilon Greedy..")
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = list(self.G.neighbors(cur))
                if len(cur_nbrs) == 0:
                    break
                if random.random() < self.alpha:  # only for exploration
                    _next = np.random.choice(cur_nbrs)
                else:
                    if self.reverse:
                        probabilities = self._get_reverse_transition_probability(cur_nbrs, memory)
                    else:
                        probabilities = self._get_transition_probability(cur_nbrs, memory)
                    # _next = np.random.choice(cur_nbrs, p=probabilities)
                    _next = cur_nbrs[probabilities.index(max(probabilities))]
                walk.append(_next)
                memory[_next] += 1

        if self.explore == "exploration":
            # print("UCB.")
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = list(self.G.neighbors(cur))
                if len(cur_nbrs) == 0:
                    break
                probabilities = self._get_all_transition_probability(cur_nbrs, memory)
                _next = np.random.choice(cur_nbrs, p=probabilities)
                # _next = cur_nbrs[probabilities.index(max(probabilities))]
                walk.append(_next)
                memory[_next] += 1
        return walk

    def _simulate_walks(self, walk_length, num_walks):
        # Repeatedly simulate random walks from each node.
        G = self.G
        nodes = list(G.nodes())
        print('node number:', len(nodes))

        # initialize occupation vector
        initial_vector = dict()
        for node in G.nodes():
            initial_vector[node] = 1

        agents = mp.cpu_count()
        pool = mp.Pool(processes=agents)
        walks = list()
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            occupation_vector = copy.copy(initial_vector)  # initialization

            nodes_zip = list(
                zip(nodes, repeat(walk_length), repeat(occupation_vector)))
            walks = walks + \
                     list(pool.starmap(self._vertex_reinforced_walk, nodes_zip))
        pool.close()
        return walks
