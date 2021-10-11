import pickle
import random
from pathlib import Path

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
from scipy import sparse

from src.utils import load_gcn_data, load_new_data, load_word_data


def _read_file(filepath, func, head=False):
    result = []
    with open(filepath) as f:
        if head:
            f.readline()
        for line in f:
            result.append(func(line.strip()))
    return result


def _read_node_int(filepath, head=False):
    return _read_file(filepath=filepath, func=lambda x: int(x), head=head)


def _read_node_str(filepath, head=False):
    return _read_file(filepath=filepath, func=lambda x: x, head=head)


def _read_edge_str(filepath, head=False):
    return _read_file(filepath=filepath, func=lambda x: x.split("\t"), head=head)


def _read_edge_int(filepath, head=False):
    return _read_file(filepath=filepath, func=lambda x: [int(tmp) for tmp in x.split("\t")], head=head)


class GraphDataset(object):
    def __init__(self, dataset_str, existing_partition=0, raw=True, root_dir="data"):
        self.dataset_str = dataset_str
        self.existing_partition = existing_partition
        self.root_dir = Path(root_dir)
        if raw:
            self._load_dataset_raw(dataset_str=dataset_str)
        else:
            self._load_dataset_pickled(dataset_path=dataset_str)

    def _load_dataset_raw(self, dataset_str):
        if dataset_str in ["cora", "citeseer", "pubmed"]:
            data = load_gcn_data(dataset_str, root_dir=self.root_dir)
            adj, features, _, _, _, train_mask, val_mask, test_mask, labels = data

            labels = np.argmax(labels, axis=-1)
            self.adj = adj
            self.G = nx.DiGraph(adj)
            self.G_self_loop = nx.DiGraph(adj + sparse.eye(adj.shape[0]))
            self.node_list = np.array(sorted(self.G.nodes()))
            self.features = features.toarray()
            self.labels = labels
            self.train_nodes = self.node_list[train_mask]
            self.val_nodes = self.node_list[val_mask]
            self.test_nodes = self.node_list[test_mask]
            output_pickle_file_name = self.root_dir / f"gcn_data/{dataset_str}.pickle.bin"

        elif dataset_str in ["chameleon", "cornell", "film", "squirrel", "texas", "wisconsin"]:
            dg, node2feature, node2label, features, labels = load_new_data(dataset_str=dataset_str,
                                                                           root_dir=self.root_dir)
            self.adj = nx.adjacency_matrix(dg, sorted(dg.nodes()))
            self.features = np.array(
                [features for _, features in sorted(dg.nodes(data='feature'), key=lambda x: x[0])])
            self.labels = np.array([label for _, label in sorted(dg.nodes(data='label'), key=lambda x: x[0])])
            self.G = dg
            self.G_self_loop = nx.DiGraph(self.adj + sparse.eye(self.adj.shape[0]))
            self.node_list = sorted(self.G.nodes())
            if self.existing_partition:
                self.train_nodes = _read_node_int(self.root_dir + f"/new_data/{self.dataset_str}/nodes.train")
                self.val_nodes = _read_node_int(self.root_dir + f"/new_data/{self.dataset_str}/nodes.val")
                self.test_nodes = _read_node_int(self.root_dir + f"/new_data/{self.dataset_str}/nodes.test")
            else:
                interest_node_ids = self.node_list.copy()
                random.shuffle(interest_node_ids)
                validation_size = int(len(interest_node_ids) * 0.1)
                test_size = int(len(interest_node_ids) * 0.1)
                self.val_nodes = interest_node_ids[:validation_size]
                self.test_nodes = interest_node_ids[validation_size:(validation_size + test_size)]
                self.train_nodes = [node_id for node_id in interest_node_ids if
                                    node_id not in self.val_nodes and node_id not in self.test_nodes]
            output_pickle_file_name = self.root_dir / f"new_data/{dataset_str}/{dataset_str}.pickle.bin"
        elif dataset_str in ["hep-small", "hep-large", "dblp", "cora_enrich"]:
            dg, dwg, bert_g,w2v_wg, jose_wg = load_word_data(dataset_str=dataset_str, root_dir=self.root_dir)
            self.adj = nx.adjacency_matrix(dg, sorted(dg.nodes()))
            self.bert_adj = nx.adjacency_matrix(bert_g, sorted(dg.nodes()))
            self.features = np.array(
                [features for _, features in sorted(dg.nodes(data='feature'), key=lambda x: x[0])])
            self.bert_features = np.array(
                [features for _, features in sorted(dg.nodes(data='bert_feature'), key=lambda x: x[0])])
            self.word_adj = nx.adjacency_matrix(dwg, sorted(dwg.nodes()))
            self.w2v_word_adj = nx.adjacency_matrix(w2v_wg, sorted(dwg.nodes()))
            self.jose_word_adj = nx.adjacency_matrix(jose_wg, sorted(dwg.nodes()))
            self.word_features = np.array(
                [features for _, features in sorted(dwg.nodes(data='feature'), key=lambda x: x[0])])
            self.word_jose_features = np.array(
                [features for _, features in sorted(dwg.nodes(data='jose_feature'), key=lambda x: x[0])])
            self.labels = np.array([label for _, label in sorted(dg.nodes(data='label'), key=lambda x: x[0])])
            self.G = dg
            self.G_self_loop = nx.Graph(self.adj + sparse.eye(self.adj.shape[0]))
            # self.WG = dwg
            # self.WG_self_loop = nx.Graph(self.word_adj + sparse.eye(self.word_adj.shape[0]))
            self.node_list = sorted(self.G.nodes())
            if self.existing_partition:
                self.train_nodes = _read_node_int(self.root_dir + f"/word_data/{self.dataset_str}/nodes.train")
                self.val_nodes = _read_node_int(self.root_dir + f"/word_data/{self.dataset_str}/nodes.val")
                self.test_nodes = _read_node_int(self.root_dir + f"/word_data/{self.dataset_str}/nodes.test")
            else:
                interest_node_ids = self.node_list.copy()
                random.shuffle(interest_node_ids)
                validation_size = int(len(interest_node_ids) * 0.1)
                test_size = int(len(interest_node_ids) * 0.1)
                self.val_nodes = interest_node_ids[:validation_size]
                self.test_nodes = interest_node_ids[validation_size:(validation_size + test_size)]
                self.train_nodes = [node_id for node_id in interest_node_ids if
                                    node_id not in self.val_nodes and node_id not in self.test_nodes]

            output_pickle_file_name = self.root_dir / f"word_data/{dataset_str}/{dataset_str}.pickle.bin"
        else:
            print(f"{dataset_str} not defined")
            raise NotImplementedError
        print("start saving pickle data")

        with open(output_pickle_file_name, 'wb') as fout:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = {
                "dataset_str": self.dataset_str,
                "adj": self.adj,
                "G": self.G,
                "G_self_loop": self.G_self_loop,
                "node_list": self.node_list,
                "train_nodes": self.train_nodes,
                "val_nodes": self.val_nodes,
                "test_nodes": self.test_nodes,
                "features": self.features,
                "labels": self.labels,
            }
            if dataset_str in ["hep-small", "hep-large", "dblp","cora_enrich"]:
                # data["WG"] = self.WG
                # data["WG_self_loop"] = self.WG_self_loop
                data["bert_adj"] = self.bert_adj
                data["jose_word_adj"] = self.jose_word_adj
                data["w2v_word_adj"] = self.w2v_word_adj
                data["word_adj"] = self.word_adj
                data["word_features"] = self.word_features
                data["jose_features"] = self.word_jose_features
                data["bert_features"] = self.bert_features
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def _load_dataset_pickled(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = pickle.load(f)
            self.dataset_str = data["dataset_str"]
            self.adj = data["adj"]
            self.G = data["G"]
            self.G_self_loop = data["G_self_loop"]
            self.node_list = data["node_list"]
            self.train_nodes = data["train_nodes"]
            self.val_nodes = data["val_nodes"]
            self.test_nodes = data["test_nodes"]
            self.features = data["features"]
            self.labels = data["labels"]
            if self.dataset_str in ["hep-small", "hep-large", "dblp","cora_enrich"]:
                # self.WG = data["WG"]
                # self.WG_self_loop = data["WG_self_loop"]
                self.bert_adj = data["bert_adj"]
                self.jose_word_adj = data["jose_word_adj"]
                self.w2v_word_adj = data["w2v_word_adj"]
                self.word_adj = data["word_adj"]
                self.word_features = data["word_features"]
                self.word_jose_features = data["jose_features"]
                self.bert_features = data["bert_features"]
        print(f"Loaded pickled dataset from {dataset_path}")


class GCNDatasetDeprecated():
    def __init__(self, graph_dataset, mode="train"):
        self.mode = mode
        self.graph_dataset = graph_dataset
        node_list = graph_dataset.node_list
        self.train_masks = [node in graph_dataset.train_nodes for node in node_list]
        self.val_masks = [node in graph_dataset.val_nodes for node in node_list]
        self.test_masks = [node in graph_dataset.test_nodes for node in node_list]

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if self.mode == "train":
            return self.train_masks
        elif self.mode == "validation":
            return self.val_masks
        elif self.mode == "test":
            return self.test_masks
        else:
            print(f"invalid mode {self.mode}")
            raise NotImplementedError


class HeteroGCNDataset(object):
    def __init__(self, graph_dataset, device=None,node_graph="raw",word_graph="raw"):
        self.graph_dataset = graph_dataset
        node_list = graph_dataset.node_list
        self.device = device if device else "cpu"
        self.labels = torch.tensor(graph_dataset.labels).long().to(self.device)
        self.train_masks = [node in graph_dataset.train_nodes for node in node_list]
        self.val_masks = [node in graph_dataset.val_nodes for node in node_list]
        self.test_masks = [node in graph_dataset.test_nodes for node in node_list]
        self.features = graph_dataset.features
        self.num_node = self.features.shape[0]
        self.num_feature = self.features.shape[1]
        self.dataset_str = graph_dataset.dataset_str
        if node_graph == "bert":
            self.adj = graph_dataset.bert_adj
        else:
            self.adj = graph_dataset.adj

        self.adj.resize((self.num_node + self.num_feature, self.num_node + self.num_feature))
        self.adj = self.adj + sparse.eye(self.num_node + self.num_feature)
        node_feature_adj = sparse.hstack(
            [sparse.csr_matrix(np.zeros((self.num_node, self.num_node))),
             sparse.csr_matrix(self.features)])
        node_feature_adj.resize((self.num_node + self.num_feature, self.num_node + self.num_feature))
        all_adj = node_feature_adj + node_feature_adj.transpose() + self.adj
        node_feature_adj = node_feature_adj + sparse.eye(self.num_node + self.num_feature)

        self.node_graph = dgl.DGLGraph(self.adj)
        self.node_feature_graph = dgl.DGLGraph(node_feature_adj)
        self.feature_node_graph = dgl.DGLGraph(node_feature_adj.transpose())

        self.ntypes = ["node", "feature"]
        self.etypes = [("node", "node"), ("feature", "node"), ("node", "feature"), ("feature", "feature")]
        if graph_dataset.dataset_str in ["hep-small", "hep-large", "dblp","cora_enrich"]:
            self.bert_features = graph_dataset.bert_features
            self.word_jose_features = graph_dataset.word_jose_features
            self.word_features = graph_dataset.word_features
            self.word_adj = sparse.csr_matrix(
                np.zeros((self.num_node + self.num_feature, self.num_node + self.num_feature)))
            if word_graph == "w2v":
                word_adj = graph_dataset.w2v_word_adj
            elif word_graph == "jose":
                word_adj = graph_dataset.jose_word_adj
            else:
                word_adj = graph_dataset.word_adj
            self.word_adj[-self.num_feature:, -self.num_feature:] = word_adj
            self.word_adj = self.word_adj + sparse.eye(self.num_node + self.num_feature)
            self.full_adj = (all_adj + self.word_adj) > 0
            self.feature_graph = dgl.DGLGraph(self.word_adj)
            self.full_graph = dgl.DGLGraph(self.full_adj)
            feature_feats = torch.tensor(self.word_features)
            word_jose_feats = torch.tensor(self.word_jose_features)
            bert_feats = torch.tensor(self.bert_features)
            empty_node_feats = torch.zeros((self.num_node, feature_feats.shape[1]))
            pseudo_node_feature_feats = torch.cat([empty_node_feats, feature_feats])
            node_feats = self.generate_node_feats(self.feature_node_graph, pseudo_node_feature_feats)
            one_hot_node_feats = torch.tensor(self.features)
            # node_feature_feats = torch.cat([node_feats, feature_feats])
        else:
            self.feature_graph = dgl.DGLGraph(sparse.eye(self.num_node + self.num_feature))
            node_feats = torch.tensor(self.features)
            feature_feats = torch.eye(self.num_feature)
            one_hot_node_feats = node_feats
            word_jose_feats = bert_feats = None
            self.full_graph = dgl.DGLGraph(all_adj)
            # node_feature_feats = torch.cat([node_feats, feature_feats])

        self.features_dict = {
            "node": node_feats,
            "one_hot_node": one_hot_node_feats,
            "feature": feature_feats,
            "word_jose": word_jose_feats,
            "word_w2v": feature_feats,
            "node_bert": bert_feats
        }
        self.graph_dict = {
            ("node", "node"): self.node_graph,
            ("node", "feature"): self.node_feature_graph,
            ("feature", "node"): self.feature_node_graph,
            ("feature", "feature"): self.feature_graph,
            "full": self.full_graph,
        }
        self.type2mask = {
            "node": torch.tensor(range(self.num_node)),
            "feature": torch.tensor(range(self.num_feature)) + self.num_node
        }

    def generate_node_feats(self, pseudo_feature_node_graph, pseudo_node_feature_feats):
        pseudo_feature_node_graph = pseudo_feature_node_graph.local_var()
        pseudo_feature_node_graph.srcdata['h'] = pseudo_node_feature_feats
        pseudo_feature_node_graph.update_all(fn.copy_src(src='h', out='m'), fn.mean(msg='m', out='h'))
        return pseudo_feature_node_graph.ndata['h'][:self.num_node]

    def get_data(self):
        return (
            self.ntypes,
            self.etypes,
            self.graph_dict,
            self.features_dict,
            self.type2mask,
            self.train_masks,
            self.val_masks,
            self.test_masks,
            self.labels
        )

    def get_masks(self):
        return self.type2mask, self.train_masks, self.val_masks, self.test_masks, self.labels

    def get_all_graphs(self):
        return [self.graph_dict[etype] for etype in self.etypes]

    def get_graph(self, etype):
        return self.graph_dict[etype]

    def get_featuers(self, mode="ont_hot"):
        node_feats, feature_feats = self.features_dict["node"], self.features_dict["feature"]
        one_hot_node_feats = self.features_dict["one_hot_node"]
        if mode == "one_hot":
            empty_node_feats = torch.zeros((one_hot_node_feats.shape[0], feature_feats.shape[1]))
            empty_feature_feats = torch.zeros((feature_feats.shape[0], one_hot_node_feats.shape[1]))
            node_dim_feats = torch.cat([one_hot_node_feats.float(), empty_feature_feats])
            feature_dim_feats = torch.cat([empty_node_feats, feature_feats.float()])
            return torch.cat([node_dim_feats, feature_dim_feats], dim=1)
        elif mode == "message_passing":
            return torch.cat([node_feats, feature_feats])


if __name__ == '__main__':
    GraphDataset(dataset_str="hep-small", root_dir="../data")
