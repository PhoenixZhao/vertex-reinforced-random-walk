import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle

from cogdl.data import Dataset, InMemoryDataset
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task

warnings.filterwarnings("ignore")


@register_task("unsupervised_node_classification")
class UnsupervisedNodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-shuffle", type=int, default=5)
        # fmt: on

    def __init__(self, args):
        super(UnsupervisedNodeClassification, self).__init__(args)
        dataset = build_dataset(args)
        self.data = dataset[0]
        if issubclass(dataset.__class__.__bases__[0], InMemoryDataset):
            self.num_nodes = self.data.y.shape[0]
            self.num_classes = dataset.num_classes
            self.label_matrix = np.zeros((self.num_nodes, self.num_classes), dtype=int)
            self.label_matrix[range(self.num_nodes), self.data.y] = 1
        else:
            self.label_matrix = self.data.y
            self.num_nodes, self.num_classes = self.data.y.shape

        self.model = build_model(args)
        self.hidden_size = args.hidden_size
        self.num_shuffle = args.num_shuffle
        self.is_weighted = self.data.edge_attr is not None

    def train(self):
        G = nx.Graph()
        if self.is_weighted:
            edges, weight = (
                self.data.edge_index.t().tolist(),
                self.data.edge_attr.tolist(),
            )
            G.add_weighted_edges_from(
                [(edges[i][0], edges[i][1], weight[0][i]) for i in range(len(edges))]
            )
        else:
            G.add_edges_from(self.data.edge_index.t().tolist())
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((self.num_nodes, self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]

        # label nor multi-label
        label_matrix = sp.csr_matrix(self.label_matrix)

        return self._evaluate(features_matrix, label_matrix, self.num_shuffle)

    def _evaluate(self, features_matrix, label_matrix, num_shuffle):
        # features_matrix, node2id = utils.load_embeddings(args.emb)
        # label_matrix = utils.load_labels(args.label, node2id, divi_str=" ")

        # shuffle, to create train/test groups
        shuffles = []
        for _ in range(num_shuffle):
            shuffles.append(skshuffle(features_matrix, label_matrix))

        # score each train/test group
        all_results_micro = defaultdict(list)
        all_results_macro = defaultdict(list)
        # training_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        training_percents = [0.1, 0.3, 0.5, 0.7, 0.9]

        for train_percent in training_percents:
            for shuf in shuffles:
                X, y = shuf

                training_size = int(train_percent * self.num_nodes)

                X_train = X[:training_size, :]
                y_train = y[:training_size, :]

                X_test = X[training_size:, :]
                y_test = y[training_size:, :]

                clf = TopKRanker(LogisticRegression())
                clf.fit(X_train, y_train)

                # find out how many labels should be predicted
                top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
                preds = clf.predict(X_test, top_k_list)
                result = f1_score(y_test, preds, average="micro")
                all_results_micro[train_percent].append(result)

                result = f1_score(y_test, preds, average="macro")
                all_results_macro[train_percent].append(result)
            # print("micro", result)

        micro_ = dict(
            (
                f"Micro-F1 {train_percent}",
                sum(all_results_micro[train_percent]) / len(all_results_micro[train_percent]),
            )
            for train_percent in sorted(all_results_micro.keys())
        )
        macro_ = dict(
            (
                f"Macro-F1 {train_percent}",
                sum(all_results_macro[train_percent]) / len(all_results_macro[train_percent]),
            )
            for train_percent in sorted(all_results_macro.keys())
        )
        micro_.update(macro_)
        return micro_


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sp.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels
