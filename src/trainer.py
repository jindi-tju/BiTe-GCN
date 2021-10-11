# node_node, node_feature, feature_node, feature_feature
import json
import os
import time

import dgl
import numpy as np
import torch
from torch.functional import F

from src.dataset import HeteroGCNDataset, GraphDataset
from src.model import HeteroGATNet
from src.model_old import HeteroGCN, VanillaGCN, VanillaGAT


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.graph_dataset = GraphDataset(dataset_str=args.dataset, raw=False)

    def train(self):
        for epoch in range(self.args.epochs):
            t0 = time.time()
            self.net.train()
            train_logits = self.net(self.g, self.features)
            train_logp = F.log_softmax(train_logits, 1)
            train_loss = F.nll_loss(train_logp[self.train_mask], self.labels[self.train_mask])
            train_pred = train_logp.argmax(dim=1)
            train_acc = torch.eq(train_pred[self.train_mask], self.labels[self.train_mask]).float().mean().item()
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(self.g, self.features)
                val_logp = F.log_softmax(val_logits, 1)
                val_loss = F.nll_loss(val_logp[self.val_mask], self.labels[self.val_mask]).item()
                val_pred = val_logp.argmax(dim=1)
                val_acc = torch.eq(val_pred[self.val_mask], self.labels[self.val_mask]).float().mean().item()

            self.learning_rate_scheduler.step(val_loss)

            self.dur.append(time.time() - t0)
            if epoch % self.args.log_interval == 0 and self.args.verbose:
                print(
                    "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                        epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(self.dur) / len(self.dur)))
            if val_acc >= vacc_mx or val_loss <= vlss_mn:
                if val_acc >= vacc_mx and val_loss <= vlss_mn:
                    state_dict_early_model = self.net.state_dict()
                vacc_mx = np.max((val_acc, vacc_mx))
                vlss_mn = np.min((val_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= self.patience:
                    break
        self.net.load_state_dict(state_dict_early_model)
        self.net.eval()
        with torch.no_grad():
            test_logits = self.net(self.g, self.features)
            test_logp = F.log_softmax(test_logits, 1)
            test_loss = F.nll_loss(test_logp[self.test_mask], self.labels[self.test_mask]).item()
            test_pred = test_logp.argmax(dim=1)
            test_acc = torch.eq(test_pred[self.test_mask], self.labels[self.test_mask]).float().mean().item()

        print("Test_acc" + ":" + str(test_acc))

        results_dict = vars(self.args)
        results_dict['test_loss'] = test_loss
        results_dict['test_acc'] = test_acc
        results_dict['actual_epochs'] = 1 + epoch
        results_dict['val_acc_max'] = vacc_mx
        results_dict['val_loss_min'] = vlss_mn
        results_dict['total_time'] = sum(self.dur)

        with open(os.path.join('logs', f'{self.args.model}_results.json'), 'w') as outfile:
            outfile.write(json.dumps(results_dict, indent=2) + '\n')


class HeteroGCNTrainer(BaseTrainer):
    def __init__(self, args):
        super(HeteroGCNTrainer, self).__init__(args)
        self.hetero_gcn_dataset = HeteroGCNDataset(graph_dataset=self.graph_dataset, device=self.device)
        self.ntypes, self.etypes, self.graph_dict, self.features_dict, self.type2mask, self.train_mask, self.val_mask, self.test_mask, self.labels = self.hetero_gcn_dataset.get_data()
        self.g = [self.graph_dict[etype] for etype in self.etypes]
        if args.word_emb == "w2v":
            self.feature_feats = self.features_dict["word_w2v"]
        elif args.word_emb == "jose":
            self.feature_feats = self.features_dict["word_jose"]
        else:
            raise NotImplementedError
        if args.node_feature == "message_passing":
            self.node_feats = self.features_dict["node"]
            self.features = torch.cat([self.node_feats, self.feature_feats]).to(self.device)
            self.in_dim = self.features.shape[1]
            self.net = HeteroGCN(in_dim=self.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                                 num_layers=args.num_layer,
                                 ntypes=self.ntypes,
                                 etypes=self.etypes,
                                 type2mask=self.type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                                 in_dropout=args.dropout_rate,
                                 hidden_dropout=args.dropout_rate, output_dropout=0.1).to(self.device)
        elif args.node_feature == "one_hot":
            self.one_hot_node_feats = self.features_dict["one_hot_node"]
            self.empty_node_feats = torch.zeros((self.one_hot_node_feats.shape[0], self.feature_feats.shape[1]))
            self.empty_feature_feats = torch.zeros((self.feature_feats.shape[0], self.one_hot_node_feats.shape[1]))
            self.node_dim_feats = torch.cat([self.one_hot_node_feats.float(), self.empty_feature_feats]).to(self.device)
            self.node_dim = self.one_hot_node_feats.shape[1]
            self.feature_dim_feats = torch.cat([self.empty_node_feats, self.feature_feats.float()]).to(self.device)
            self.feature_dim = self.feature_feats.shape[1]
            self.features = [self.node_dim_feats, self.node_dim_feats, self.feature_dim_feats, self.feature_dim_feats]
            self.in_dims = [self.node_dim, self.node_dim, self.feature_dim, self.feature_dim]
            self.hidden_dims = args.hidden_dim
            self.out_dims = args.out_dim
            self.net = HeteroGCN(in_dim=self.in_dims, hidden_dim=self.hidden_dims, out_dim=self.out_dims,
                                 num_layers=args.num_layer,
                                 ntypes=self.ntypes,
                                 etypes=self.etypes,
                                 type2mask=self.type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                                 in_dropout=args.dropout_rate,
                                 hidden_dropout=args.dropout_rate, output_dropout=0.1).to(self.device)
        elif args.node_feature == "bert":
            self.bert_feats = self.features_dict["node_bert"]
            self.empty_node_feats = torch.zeros((self.bert_feats.shape[0], self.feature_feats.shape[1]))
            self.empty_feature_feats = torch.zeros((self.feature_feats.shape[0], self.bert_feats.shape[1]))
            self.node_dim_feats = torch.cat([self.bert_feats.float(), self.empty_feature_feats]).to(self.device)
            self.node_dim = self.bert_feats.shape[1]
            self.feature_dim_feats = torch.cat([self.empty_node_feats, self.feature_feats.float()]).to(self.device)
            self.feature_dim = self.feature_feats.shape[1]
            self.features = [self.node_dim_feats, self.node_dim_feats, self.feature_dim_feats, self.feature_dim_feats]
            self.in_dims = [self.node_dim, self.node_dim, self.feature_dim, self.feature_dim]
            self.net = HeteroGCN(in_dim=self.in_dims, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                                 num_layers=args.num_layer,
                                 ntypes=self.ntypes,
                                 etypes=self.etypes,
                                 type2mask=self.type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                                 in_dropout=args.dropout_rate,
                                 hidden_dropout=args.dropout_rate, output_dropout=0.1).to(self.device)
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                                  factor=args.learning_rate_decay_factor,
                                                                                  patience=args.learning_rate_decay_patience,
                                                                                  verbose=args.verbose)
        self.patience = args.patience
        self.vlss_mn = np.inf
        self.vacc_mx = 0.0
        self.state_dict_early_model = None
        self.curr_step = 0
        self.dur = []


def hetero_gcn(args):
    device = torch.device(args.device)
    graph_dataset = GraphDataset(dataset_str=args.dataset, raw=False)
    hetero_gcn_dataset = HeteroGCNDataset(graph_dataset=graph_dataset, device=device,word_graph = args.word_feature,node_graph=args.node_feature)
    ntypes, etypes, graph_dict, features_dict, type2mask, train_mask, val_mask, test_mask, labels = hetero_gcn_dataset.get_data()
    g = [graph_dict[etype] for etype in etypes]
    if args.word_feature == "w2v":
        feature_feats = features_dict["word_w2v"]
    elif args.word_feature == "jose":
        feature_feats = features_dict["word_jose"]
    else:
        feature_feats = features_dict["feature"]
    if args.node_feature == "message_passing":
        node_feats = features_dict["node"]
        features = torch.cat([node_feats, feature_feats]).to(device)
        in_dim = features.shape[1]
        net = HeteroGCN(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, num_layers=args.num_layer,
                        ntypes=ntypes,
                        etypes=etypes,
                        type2mask=type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                        in_dropout=args.dropout_rate,
                        hidden_dropout=args.dropout_rate, output_dropout=0.1).to(device)
    elif args.node_feature == "one_hot" or args.node_feature == "raw":
        one_hot_node_feats = features_dict["one_hot_node"]
        empty_node_feats = torch.zeros((one_hot_node_feats.shape[0], feature_feats.shape[1]))
        empty_feature_feats = torch.zeros((feature_feats.shape[0], one_hot_node_feats.shape[1]))
        node_dim_feats = torch.cat([one_hot_node_feats.float(), empty_feature_feats]).to(device)
        node_dim = one_hot_node_feats.shape[1]
        feature_dim_feats = torch.cat([empty_node_feats, feature_feats.float()]).to(device)
        feature_dim = feature_feats.shape[1]
        features = [node_dim_feats, node_dim_feats, feature_dim_feats, feature_dim_feats]
        in_dims = [node_dim, node_dim, feature_dim, feature_dim]
        hidden_dims = args.hidden_dim
        out_dims = args.out_dim
        net = HeteroGCN(in_dim=in_dims, hidden_dim=hidden_dims, out_dim=out_dims, num_layers=args.num_layer,
                        ntypes=ntypes,
                        etypes=etypes,
                        type2mask=type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                        in_dropout=args.dropout_rate,
                        hidden_dropout=args.dropout_rate, output_dropout=0.1).to(device)
    elif args.node_feature == "bert":
        bert_feats = features_dict["node_bert"]
        empty_node_feats = torch.zeros((bert_feats.shape[0], feature_feats.shape[1]))
        empty_feature_feats = torch.zeros((feature_feats.shape[0], bert_feats.shape[1]))
        node_dim_feats = torch.cat([bert_feats.float(), empty_feature_feats]).to(device)
        node_dim = bert_feats.shape[1]
        feature_dim_feats = torch.cat([empty_node_feats, feature_feats.float()]).to(device)
        feature_dim = feature_feats.shape[1]
        features = [node_dim_feats, node_dim_feats, feature_dim_feats, feature_dim_feats]
        in_dims = [node_dim, node_dim, feature_dim, feature_dim]
        net = HeteroGCN(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=args.out_dim, num_layers=args.num_layer,
                        ntypes=ntypes,
                        etypes=etypes,
                        type2mask=type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                        in_dropout=args.dropout_rate,
                        hidden_dropout=args.dropout_rate, output_dropout=0.1).to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                         factor=args.learning_rate_decay_factor,
                                                                         patience=args.learning_rate_decay_patience,
                                                                         verbose=args.verbose)
    patience = args.patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    dur = []
    for epoch in range(args.epochs):
        t0 = time.time()

        net.train()
        train_logits = net(g, features)
        train_logp = F.log_softmax(train_logits[type2mask["node"]], 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_logits = net(g, features)
            val_logp = F.log_softmax(val_logits[type2mask["node"]], 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)
        if epoch % args.log_interval == 0 and args.verbose:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break
    net.load_state_dict(state_dict_early_model)
    net.eval()
    with torch.no_grad():
        test_logits = net(g, features)
        test_logp = F.log_softmax(test_logits[type2mask["node"]], 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = torch.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()

    print("Test_acc" + ":" + str(test_acc))

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)

    with open(os.path.join('logs', f'{args.model}_results.json'), 'w') as outfile:
        outfile.write(json.dumps(results_dict, indent=2) + '\n')


def hetero_gat(args):
    device = torch.device(args.device)
    graph_dataset = GraphDataset(dataset_str=args.dataset, raw=False)
    hetero_gcn_dataset = HeteroGCNDataset(graph_dataset=graph_dataset, device=device)
    ntypes, etypes, graph_dict, features_dict, type2mask, train_mask, val_mask, test_mask, labels = hetero_gcn_dataset.get_data()

    g = [graph_dict[etype] for etype in etypes]
    if args.word_feature == "w2v":
        feature_feats = features_dict["word_w2v"]
    elif args.word_feature == "jose":
        feature_feats = features_dict["word_jose"]
    else:
        raise NotImplementedError
    full_graph = graph_dict["full"]
    if args.node_feature == "message_passing":
        node_feats = features_dict["node"]
        features = torch.cat([node_feats, feature_feats]).to(device)
        in_dim = features.shape[1]
    else:
        if args.node_feature == "one_hot" or args.node_feature == "raw":
            one_hot_node_feats = features_dict["one_hot_node"]
            empty_node_feats = torch.zeros((one_hot_node_feats.shape[0], feature_feats.shape[1]))
            empty_feature_feats = torch.zeros((feature_feats.shape[0], one_hot_node_feats.shape[1]))
            node_dim_feats = torch.cat([one_hot_node_feats.float(), empty_feature_feats])
            node_dim = one_hot_node_feats.shape[1]
            feature_dim_feats = torch.cat([empty_node_feats, feature_feats.float()])
            feature_dim = feature_feats.shape[1]
            features = torch.cat((node_dim_feats, feature_dim_feats), dim=1).to(device)
            in_dim = node_dim + feature_dim
        elif args.node_feature == "bert":
            bert_feats = features_dict["node_bert"]
            empty_node_feats = torch.zeros((bert_feats.shape[0], feature_feats.shape[1]))
            empty_feature_feats = torch.zeros((feature_feats.shape[0], bert_feats.shape[1]))
            node_dim_feats = torch.cat([bert_feats.float(), empty_feature_feats])
            node_dim = bert_feats.shape[1]
            feature_dim_feats = torch.cat([empty_node_feats, feature_feats.float()])
            feature_dim = feature_feats.shape[1]
            features = torch.cat((node_dim_feats, feature_dim_feats), dim=1).to(device)
            in_dim = node_dim + feature_dim
        else:
            raise NotImplementedError
    net = HeteroGATNet(g=full_graph, num_input_features=in_dim, num_output_classes=args.out_dim,
                       num_hidden=args.hidden_dim, type2mask=type2mask,
                       num_heads_layer_one=args.num_head, num_heads_layer_two=args.num_head, residual=args.residual,
                       dropout_rate=args.dropout_rate, num_layers=args.num_layer).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                         factor=args.learning_rate_decay_factor,
                                                                         patience=args.learning_rate_decay_patience,
                                                                         verbose=args.verbose)
    patience = args.patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    dur = []
    for epoch in range(args.epochs):
        t0 = time.time()

        net.train()
        train_logits = net(g, features)
        train_logp = F.log_softmax(train_logits[type2mask["node"]], 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_logits = net(g, features)
            val_logp = F.log_softmax(val_logits[type2mask["node"]], 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)
        if epoch % args.log_interval == 0 and args.verbose:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break
    net.load_state_dict(state_dict_early_model)
    net.eval()
    with torch.no_grad():
        test_logits = net(g, features)
        test_logp = F.log_softmax(test_logits[type2mask["node"]], 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = torch.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()

    print("Test_acc" + ":" + str(test_acc))

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)

    with open(os.path.join('logs', f'{args.model}_results.json'), 'w') as outfile:
        outfile.write(json.dumps(results_dict, indent=2) + '\n')


def vanilla_gcn(args):
    device = torch.device(args.device)
    graph_dataset = GraphDataset(dataset_str=args.dataset, raw=False)

    train_mask = [node in graph_dataset.train_nodes for node in graph_dataset.node_list]
    val_mask = [node in graph_dataset.val_nodes for node in graph_dataset.node_list]
    test_mask = [node in graph_dataset.test_nodes for node in graph_dataset.node_list]
    labels = torch.tensor(graph_dataset.labels).to(device)
    g = dgl.DGLGraph(graph_dataset.G_self_loop)
    features = torch.tensor(graph_dataset.features).float().to(device)
    if args.node_feature == "raw":
        features = torch.eye(features.shape[0]).float().to(device)
        # features = np.loadtxt(f"/mnt/data/Ubuntu/charms/BiGCN/data/word_data/{args.dataset_str}/gcn_features.txt")
        # features = torch.tensor(features).float().to(device)
    in_dim = features.shape[1]
    net = VanillaGCN(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, num_layers=args.num_layer,
                     activation=F.leaky_relu, in_dropout=0.1,
                     hidden_dropout=args.dropout_rate,
                     output_dropout=0.1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                         factor=args.learning_rate_decay_factor,
                                                                         patience=args.learning_rate_decay_patience,
                                                                         verbose=args.verbose)
    patience = args.patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    dur = []
    for epoch in range(args.epochs):
        t0 = time.time()

        net.train()
        train_logits = net(g, features)
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_logits = net(g, features)
            val_logp = F.log_softmax(val_logits, 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)
        if epoch % args.log_interval == 0 and args.verbose:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break
    net.load_state_dict(state_dict_early_model)
    net.eval()
    with torch.no_grad():
        test_logits = net(g, features)
        test_logp = F.log_softmax(test_logits, 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = torch.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()

    print("Test_acc" + ":" + str(test_acc))

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)

    with open(os.path.join('logs', f'{args.model}_results.json'), 'w') as outfile:
        outfile.write(json.dumps(results_dict, indent=2) + '\n')


def vanilla_gat(args):
    device = torch.device(args.device)
    graph_dataset = GraphDataset(dataset_str=args.dataset, raw=False)

    train_mask = [node in graph_dataset.train_nodes for node in graph_dataset.node_list]
    val_mask = [node in graph_dataset.val_nodes for node in graph_dataset.node_list]
    test_mask = [node in graph_dataset.test_nodes for node in graph_dataset.node_list]
    labels = torch.tensor(graph_dataset.labels).to(device)
    g = dgl.DGLGraph(graph_dataset.G_self_loop)
    features = torch.tensor(graph_dataset.features).float().to(device)
    in_dim = features.shape[1]
    net = VanillaGAT(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, num_layers=args.num_layer,
                     activation=F.leaky_relu, residual=args.residual, feat_drop=args.dropout_rate,
                     heads=[args.num_head] * (1 + args.num_layer), attn_drop=args.dropout_rate).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                         factor=args.learning_rate_decay_factor,
                                                                         patience=args.learning_rate_decay_patience,
                                                                         verbose=args.verbose)
    patience = args.patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    dur = []
    for epoch in range(args.epochs):
        t0 = time.time()

        net.train()
        train_logits = net(g, features)
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_logits = net(g, features)
            val_logp = F.log_softmax(val_logits, 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)
        if epoch % args.log_interval == 0 and args.verbose:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break
    net.load_state_dict(state_dict_early_model)
    net.eval()
    with torch.no_grad():
        test_logits = net(g, features)
        test_logp = F.log_softmax(test_logits, 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = torch.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()

    print("Test_acc" + ":" + str(test_acc))

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)

    with open(os.path.join('logs', f'{args.model}_results.json'), 'w') as outfile:
        outfile.write(json.dumps(results_dict, indent=2) + '\n')
