import numpy as np
from gensim.models import Word2Vec
from . import BaseModel, register_model
import multiprocessing as mp
from itertools import repeat
import random
import copy
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.special import softmax
from scipy.stats import wasserstein_distance
from tqdm import tqdm


@register_model("drrw")
class DRRW(BaseModel):
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
            args.div,
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
            div,
            explore):
        super(DRRW, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration
        self.alpha = alpha
        self.div = div
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

    def _get_exploration_score(self, path, y):
        # path = path.append(y)
        T_s = path.count(path[0]) + 1
        T_cur = path.count(y) + 1
        return np.sqrt(np.log(T_s) / T_cur)

    """
    def _get_exploration_score(self, path, y):
        t = len(path)
        T_cur = path.count(y) + 1
        theta = 1 + t * (np.log(t) ** 2)
        return np.sqrt(2 / T_cur * np.log(theta))
    """

    def _get_exploitation_score(self, memory, node):
        next_memory = copy.deepcopy(memory)
        node_index = memory["id"].index(node)
        next_memory["times"][node_index] += 1

        if node == -1:
            k = node_index + 1
        else:
            k = memory["id"].index(-1)
        # print("cur times:", memory["times"][:k])
        # print("next times:", next_memory["times"][:k])

        # normalized
        cur_w = np.array(memory["times"][:k]) / sum(memory["times"][:k])
        next_w = np.array(next_memory["times"][:k]) / sum(next_memory["times"][:k])
        #cur_w = softmax(memory["times"][:k])
        #next_w = softmax(next_memory["times"][:k])
        # print("cur_w", cur_w)
        # print("next_w", next_w)
        """
        if node in memory["id"]:
            node_index = memory["id"].index(node)
        else:
            node_index = memory["id"].index(-1)
        next_memory["times"][node_index] += 1
        """

        if self.div == "js":
            q_score = 1 - distance.jensenshannon(cur_w, next_w)
        elif self.div == "ws":
            q_score = 1 - wasserstein_distance(cur_w, next_w)
        else:
            q_score = 1 - entropy(cur_w, next_w)
        # print("q_score:", q_score)
        return q_score

    def _get_next_node_ucb(self, nbrs, memory, cur_walk):
        candidate_nbrs = []
        scores = []
        top_k = 10
        if len(nbrs) > top_k:
            # non occur node
            q_score = self._get_exploitation_score(memory, -1)
            u_score = self._get_exploration_score(cur_walk, -1)
            score = q_score + self.alpha * u_score

            # If the nbr have not occurred in the path
            non_occur_nodes = list(set(nbrs) - set(cur_walk))
            if len(non_occur_nodes) > top_k:
                candidate_nbrs.extend(np.random.choice(non_occur_nodes, size=top_k).tolist())
                scores.extend([score] * top_k)
            else:
                candidate_nbrs.extend(non_occur_nodes)
                scores.extend([score] * len(non_occur_nodes))

            # If the nbr have occurred in the path
            occur_nodes = list(set(nbrs).intersection(set(cur_walk)))
            for node in occur_nodes:
                q_score = self._get_exploitation_score(memory, node)
                u_score = self._get_exploration_score(cur_walk, node)
                score = q_score + self.alpha * u_score
                candidate_nbrs.append(node)
                scores.append(score)

            # arr_candidates_nbrs = np.array(candidate_nbrs)
            # arr_scores = np.array(scores)
            # indices = arr_scores.argsort()[-top_k:][::-1]
            # scores = arr_scores[indices].tolist()
            # candidate_nbrs = arr_candidates_nbrs[indices].tolist()
        else:
            # non occur node
            non_occur_nodes = list(set(nbrs) - set(cur_walk))
            if len(non_occur_nodes) > 0:
                node = np.random.choice(non_occur_nodes)
                q_score = self._get_exploitation_score(memory, -1)
                u_score = self._get_exploitation_score(memory, -1)
                score = q_score + self.alpha * u_score
                candidate_nbrs.append(node)
                scores.append(score)

            # If the nbr have occurred in the path
            occur_nodes = list(set(nbrs).intersection(set(cur_walk)))

            for node in occur_nodes:
                q_score = self._get_exploitation_score(memory, node)
                u_score = self._get_exploration_score(cur_walk, node)
                score = q_score + self.alpha * u_score
                candidate_nbrs.append(node)
                scores.append(score)

        # v = np.array(scores)
        # scores = (v - v.min() + 1e-10) / (v.max() - v.min() + 1e-10)
        probs = softmax(scores)
        next_node = np.random.choice(candidate_nbrs, p=probs)
        return next_node

    def _get_next_node(self, nbrs, memory, cur_walk):
        # initialization: suppose all the nbrs haven't appeared in the path
        candidate_nbrs = []
        q_scores = []

        # non occur node
        non_occur_nodes = list(set(nbrs) - set(cur_walk))
        if len(non_occur_nodes) > 0:
            node = np.random.choice(non_occur_nodes)
            q_score = self._get_exploitation_score(memory, -1)
            candidate_nbrs.append(node)
            q_scores.append(q_score)

        # If the nbr have occurred in the path
        occur_nodes = list(set(nbrs).intersection(set(cur_walk)))

        for node in occur_nodes:
            q_score = self._get_exploitation_score(memory, node)
            candidate_nbrs.append(node)
            q_scores.append(q_score)

        # print("q_scores:", q_scores)

        # v = np.array(q_scores)
        # q_scores = (v - v.min() + 1e-10) / (v.max() - v.min() + 1e-10)

        # print("u_scores:", u_scores)
        # probs = softmax(q_scores)
        next_node = candidate_nbrs[q_scores.index(max(q_scores))]
        # print("probs:", probs)
        return next_node

    def _vertex_reinforced_walk(self, start_node, walk_length):
        # Simulate a random walk starting from start node.

        # initialize occupation vector
        memory = dict()
        memory["times"] = [0] * walk_length
        memory["id"] = [-1] * walk_length

        walk = [start_node]
        memory["times"][0] += 1
        memory["id"][0] = start_node

        if self.explore == "decay-greedy":
            # print("Epsilon Greedy..")
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = list(self.G.neighbors(cur))
                if len(cur_nbrs) == 0:
                    break
                if random.random() < self.alpha:  # only for exploration
                    _next = np.random.choice(cur_nbrs)
                else:
                    _next = self._get_next_node(cur_nbrs, memory, walk)
                walk.append(_next)
                if _next in memory["id"]:
                    node_index = memory["id"].index(_next)
                else:
                    node_index = memory["id"].index(-1)
                    memory["id"][node_index] = _next
                memory["times"][node_index] += 1
        if self.explore == "exploration":
            # print("UCB.")
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = list(self.G.neighbors(cur))
                if len(cur_nbrs) == 0:
                    break
                _next = self._get_next_node_ucb(cur_nbrs, memory, walk)
                # _next = np.random.choice(cur_nbrs, p=probabilities)

                walk.append(_next)
                if _next in memory["id"]:
                    node_index = memory["id"].index(_next)
                else:
                    node_index = memory["id"].index(-1)
                    memory["id"][node_index] = _next
                memory["times"][node_index] += 1
                # print("memory:", memory)
        """
        else:
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = list(self.G.neighbors(cur))
                if len(cur_nbrs) == 0:
                    break
                if len(walk) < 10:  # cold start: deepwalk for the first 10 steps
                    _next = np.random.choice(cur_nbrs)
                else:
                    _next = self._get_next_node(cur_nbrs, memory, walk)
                walk.append(_next)
                if _next in memory["id"]:
                    node_index = memory["id"].index(_next)
                else:
                    node_index = memory["id"].index(-1)
                    memory["id"][node_index] = _next
                memory["times"][node_index] += 1
                # print("walk:", walk)
        """
        # import pdb
        # pdb.set_trace()
        # print("****************************************")
        return walk

    def _simulate_walks(self, walk_length, num_walks):
        # Repeatedly simulate random walks from each node.
        G = self.G
        nodes = list(G.nodes())
        print('node number:', len(nodes))

        agents = mp.cpu_count()
        pool = mp.Pool(processes=agents)
        walks = list()
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            nodes_zip = list(
                zip(nodes, repeat(walk_length)))
            walks = walks + \
                list(pool.starmap(self._vertex_reinforced_walk, nodes_zip))
        pool.close()
        return walks
