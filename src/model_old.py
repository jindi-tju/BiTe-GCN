from collections import defaultdict
from functools import partial

import dgl.function as fn
import torch as torch
import torch.nn as nn
from dgl import DGLError
from dgl import utils
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity

""" 
Graph Propagation Modules: GCN, GAT, PGCN, PGAT
"""


class GCNLayer(nn.Module):
    r"""
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 dropout,
                 activation=None,
                 norm='both',
                 weight=True,
                 bias=True,
                 ):
        super(GCNLayer, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self._activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()
        feat = self.dropout(feat)
        if self._norm == 'both':
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = torch.matmul(feat, weight)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
        else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            if weight is not None:
                rst = torch.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class GATLayer(nn.Module):
    r"""
    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads=1,
                 feat_drop=0.5,
                 attn_drop=0.5,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 ):
        super(GATLayer, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = utils.expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:  # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class VanillaGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation, in_dropout=0.1, hidden_dropout=0.1,
                 output_dropout=0.0):
        super(VanillaGCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats=in_dim, out_feats=hidden_dim, activation=activation, dropout=in_dropout))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(
                GCNLayer(in_feats=hidden_dim, out_feats=hidden_dim, activation=activation, dropout=hidden_dropout))
        # output layer
        self.layers.append(
            GCNLayer(in_feats=hidden_dim, out_feats=out_dim, activation=None, dropout=output_dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class VanillaGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads, activation, feat_drop=0.5, attn_drop=0.5,
                 leaky_relu_alpha=0.2, residual=False):
        super(VanillaGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input layer, no residual
        self.gat_layers.append(GATLayer(in_feats=in_dim, out_feats=hidden_dim, num_heads=heads[0], feat_drop=feat_drop,
                                        attn_drop=attn_drop, negative_slope=leaky_relu_alpha, residual=False))
        # hidden layers, due to multi-head, the in_dim = hidden_dim * num_heads
        for l in range(1, num_layers):
            self.gat_layers.append(
                GATLayer(in_feats=hidden_dim * heads[l - 1], out_feats=hidden_dim, num_heads=heads[l],
                         feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=leaky_relu_alpha, residual=residual))
        # output layer
        self.gat_layers.append(
            GATLayer(in_feats=hidden_dim * heads[-2], out_feats=out_dim, num_heads=heads[-1], feat_drop=feat_drop,
                     attn_drop=attn_drop, negative_slope=leaky_relu_alpha, residual=residual))

    def forward(self, g, features):
        h = features
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.activation(h)
        # output projection
        h = self.gat_layers[-1](g, h).mean(1)
        return h


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation=None, hidden_num=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)] * (hidden_num - 1))
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.fc2(x)
        if self.activation:
            x = self.activation(x)
        return x


class MultiHeadGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, num_heads, dropout_prob, merge):
        super(MultiHeadGCNLayer, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(GCNLayer(in_feats, out_feats, activation=activation, dropout=dropout_prob))
        self.merge = merge
        if merge == "linear":
            self.merge_layer = nn.Linear(num_heads * out_feats, out_feats)
        elif merge == "mlp":
            self.merge_layer = MLP(num_heads * out_feats, 2 * num_heads * out_feats, out_feats)

    def forward(self, g, feature):
        all_attention_head_outputs = [head(g, feature) for head in self.attention_heads]
        if self.merge == 'cat':
            return torch.cat(all_attention_head_outputs, dim=1)
        elif self.merge == 'mean':
            return torch.mean(torch.stack(all_attention_head_outputs), dim=0)
        else:
            return self.merge_layer(torch.cat(all_attention_head_outputs, dim=1))


class HeteroGCNLayer(nn.Module):
    """
    :param ntypes: [t1, t2, t3 ... ]
    :param etypes: [(t1, t2), (t1, t3) ...]
    """

    def __init__(self, in_feats, out_feats, ntypes, etypes, type2mask, aggr_func="linear", activation=None,
                 dropout_prob=0.0):
        super(HeteroGCNLayer, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes
        self.type2mask = type2mask
        self.gcn_layers = nn.ModuleList()
        self.multi_features = False
        if isinstance(in_feats, list):
            self.multi_features = True
            assert len(in_feats) == len(etypes)
            for in_feat, etype in zip(in_feats, etypes):
                self.gcn_layers.append(
                    GCNLayer(in_feats=in_feat, out_feats=out_feats, activation=activation, dropout=dropout_prob)
                )
        else:
            for _ in self.etypes:
                self.gcn_layers.append(
                    GCNLayer(in_feats=in_feats, out_feats=out_feats, activation=activation, dropout=dropout_prob)
                )
        if aggr_func == "mean":
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: sum(feat_l) / len(feat_l))
        elif aggr_func == "sum":
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: sum(feat_l))
        elif aggr_func == "mlp":
            self.merge = MLP(in_dim=out_feats * (len(etypes) // len(ntypes)), out_dim=out_feats, activation=activation,
                             hidden_dim=out_feats * len(etypes), hidden_num=2)
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: self.merge(torch.cat(feat_l, dim=1)))
        elif aggr_func == "linear":
            self.merge = nn.Linear(out_feats * (len(etypes) // len(ntypes)), out_feats)
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: self.merge(torch.cat(feat_l, dim=1)))
        else:
            print(f"invalid aggr_func: {aggr_func}")

    def forward(self, gs, features):
        features_dict = defaultdict(list)
        if self.multi_features:
            for g, gcn_layer, etype, h in zip(gs, self.gcn_layers, self.etypes, features):
                src_type, dst_type = etype
                h = gcn_layer(g, h)
                features_dict[dst_type].append(h)
            res = self.aggr_func(features_dict)
        else:
            for g, gcn_layer, etype in zip(gs, self.gcn_layers, self.etypes, ):
                src_type, dst_type = etype
                h = features
                h = gcn_layer(g, h)
                features_dict[dst_type].append(h)
            res = self.aggr_func(features_dict)
        return res

    def base_aggr_func(self, features_dict, aggr):
        features = None
        for t, feat_l in features_dict.items():
            feat_t = aggr(feat_l)
            if features is not None:
                features[self.type2mask[t]] = feat_t[self.type2mask[t]]
            else:
                features = feat_t
        return features


class HeteroMultiHeadGCNLayer(nn.Module):
    """
    :param ntypes: [t1, t2, t3 ... ]
    :param etypes: [(t1, t2), (t1, t3) ...]
    """

    def __init__(self, in_feats, out_feats, ntypes, etypes, type2mask, num_heads, merge='mlp', aggr_func="linear",
                 activation=None, dropout_prob=0.0):
        super(HeteroMultiHeadGCNLayer, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes
        self.type2mask = type2mask
        self.gcn_layers = nn.ModuleList()
        self.multi_features = False
        if isinstance(in_feats, list):
            self.multi_features = True
            assert len(in_feats) == len(etypes)
            for in_feat, etype in zip(in_feats, etypes):
                self.gcn_layers.append(
                    MultiHeadGCNLayer(in_feats=in_feat, out_feats=out_feats, activation=activation,
                                      dropout_prob=dropout_prob, merge=merge, num_heads=num_heads)
                )
        else:
            for _ in self.etypes:
                self.gcn_layers.append(
                    MultiHeadGCNLayer(in_feats=in_feats, out_feats=out_feats, activation=activation,
                                      dropout_prob=dropout_prob, merge=merge, num_heads=num_heads)
                )
        if aggr_func == "mean":
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: sum(feat_l) / len(feat_l))
        elif aggr_func == "sum":
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: sum(feat_l))
        elif aggr_func == "mlp":
            self.merge = MLP(in_dim=out_feats * (len(etypes) // len(ntypes)), out_dim=out_feats, activation=activation,
                             hidden_dim=out_feats * len(etypes), hidden_num=1)
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: self.merge(torch.cat(feat_l, dim=1)))
        elif aggr_func == "linear":
            self.merge = nn.Linear(out_feats * (len(etypes) // len(ntypes)), out_feats)
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: self.merge(torch.cat(feat_l, dim=1)))
        elif aggr_func == "attention":
            self.Q = nn.Linear(out_feats, num_heads * out_feats)
            self.K = nn.Linear(out_feats, num_heads * out_feats)
            self.V = nn.Linear(out_feats, num_heads * out_feats)
            self.merge = nn.MultiheadAttention(num_heads * out_feats, num_heads, dropout=dropout_prob)
            self.head_trans = nn.Linear(num_heads * out_feats, out_feats)
            self.aggr_func = partial(self.base_aggr_func,
                                     aggr=lambda feat_l: self.merge(self.Q(torch.stack(feat_l, dim=0)),
                                                                    self.K(torch.stack(feat_l, dim=0)),
                                                                    self.V(torch.stack(feat_l, dim=0))))
        else:
            print(f"invalid aggr_func: {aggr_func}")

    def forward(self, gs, features):
        features_dict = defaultdict(list)
        if self.multi_features:
            for g, gcn_layer, etype, h in zip(gs, self.gcn_layers, self.etypes, features):
                src_type, dst_type = etype
                h = gcn_layer(g, h)
                features_dict[dst_type].append(h)
            res = self.aggr_func(features_dict)
        else:
            for g, gcn_layer, etype in zip(gs, self.gcn_layers, self.etypes, ):
                src_type, dst_type = etype
                h = features
                h = gcn_layer(g, h)
                features_dict[dst_type].append(h)
            res = self.aggr_func(features_dict)
        return res

    def base_aggr_func(self, features_dict, aggr):
        features = None
        for t, feat_l in features_dict.items():
            feat_t = aggr(feat_l)
            if isinstance(feat_t, tuple):
                feat_t = self.head_trans(feat_t[0].mean(0))
            if features is not None:
                features[self.type2mask[t]] = feat_t[self.type2mask[t]]
            else:
                features = feat_t
        return features



class HeteroGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, ntypes, etypes, type2mask, heads, aggr_func="linear", activation=None,
                 feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, residual=False):
        super(HeteroGATLayer, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes
        self.type2mask = type2mask
        self.gcn_layers = nn.ModuleList()
        self.multi_features = False
        if isinstance(in_feats, list):
            self.multi_features = True
            assert len(in_feats) == len(etypes)
            for in_feat, etype in zip(in_feats, etypes):
                self.gcn_layers.append(
                    GATLayer(in_feats=in_feat, out_feats=out_feats, activation=activation, num_heads=heads,
                             feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=leaky_relu_alpha,
                             residual=residual)
                )
        else:
            for _ in self.etypes:
                self.gcn_layers.append(
                    GATLayer(in_feats=in_feats, out_feats=out_feats, activation=activation, num_heads=heads,
                             feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=leaky_relu_alpha,
                             residual=residual)
                )
        if aggr_func == "mean":
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: sum(feat_l) / len(feat_l))
        elif aggr_func == "sum":
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: sum(feat_l))
        elif aggr_func == "mlp":
            self.merge = MLP(in_dim=out_feats * (len(etypes) // len(ntypes)) * heads, out_dim=out_feats,
                             activation=activation,
                             hidden_dim=out_feats * len(etypes) * heads, hidden_num=1)
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: self.merge(torch.cat(feat_l, dim=1)))
        elif aggr_func == "linear":
            self.merge = nn.Linear(out_feats * (len(etypes) // len(ntypes)) * heads, out_feats)
            self.aggr_func = partial(self.base_aggr_func, aggr=lambda feat_l: self.merge(torch.cat(feat_l, dim=1)))
        else:
            print(f"invalid aggr_func: {aggr_func}")

    def forward(self, gs, features):
        features_dict = defaultdict(list)
        if self.multi_features:
            for g, gcn_layer, etype, h in zip(gs, self.gcn_layers, self.etypes, features):
                src_type, dst_type = etype
                h = gcn_layer(g, h).flatten(1)
                features_dict[dst_type].append(h)
            res = self.aggr_func(features_dict)
        else:
            for g, gcn_layer, etype in zip(gs, self.gcn_layers, self.etypes, ):
                src_type, dst_type = etype
                h = features
                h = gcn_layer(g, h).flatten(1)
                features_dict[dst_type].append(h)
            res = self.aggr_func(features_dict)
        return res

    def base_aggr_func(self, features_dict, aggr):
        features = None
        for t, feat_l in features_dict.items():
            feat_t = aggr(feat_l)
            if features is not None:
                features[self.type2mask[t]] = feat_t[self.type2mask[t]]
            else:
                features = feat_t
        return features


class HeteroGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, ntypes, etypes, type2mask,
                 activation=None, aggr_func="linear", in_dropout=0.1, hidden_dropout=0.1,
                 output_dropout=0.0):
        super(HeteroGCN, self).__init__()
        self.multi_features = isinstance(in_dim, list)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            HeteroGCNLayer(in_feats=in_dim, out_feats=hidden_dim, ntypes=ntypes, etypes=etypes, type2mask=type2mask,
                           aggr_func=aggr_func, activation=activation, dropout_prob=in_dropout))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(
                HeteroGCNLayer(in_feats=hidden_dim, out_feats=hidden_dim, ntypes=ntypes, etypes=etypes,
                               type2mask=type2mask,
                               aggr_func=aggr_func, activation=activation, dropout_prob=hidden_dropout))
        # output layer
        self.layers.append(
            HeteroGCNLayer(in_feats=hidden_dim, out_feats=out_dim, ntypes=ntypes, etypes=etypes, type2mask=type2mask,
                           aggr_func=aggr_func, activation=None, dropout_prob=output_dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class HeteroGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, ntypes, etypes, type2mask, heads,
                 activation=None, aggr_func="linear", residual=False, feat_drop=0.5, attn_drop=0.5,
                 leaky_relu_alpha=0.2):
        super(HeteroGAT, self).__init__()
        self.multi_features = isinstance(in_dim, list)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            HeteroGATLayer(in_feats=in_dim, out_feats=hidden_dim, ntypes=ntypes, etypes=etypes, type2mask=type2mask,
                           aggr_func=aggr_func, activation=activation, heads=heads[0], residual=False,
                           feat_drop=feat_drop, attn_drop=attn_drop, leaky_relu_alpha=leaky_relu_alpha))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(
                HeteroGATLayer(in_feats=hidden_dim, out_feats=hidden_dim, ntypes=ntypes, etypes=etypes,
                               type2mask=type2mask,
                               aggr_func=aggr_func, activation=activation, heads=heads[l + 1], residual=residual,
                               feat_drop=feat_drop, attn_drop=attn_drop, leaky_relu_alpha=leaky_relu_alpha))
        # output layer
        self.layers.append(
            HeteroGATLayer(in_feats=hidden_dim, out_feats=out_dim, ntypes=ntypes, etypes=etypes, type2mask=type2mask,
                           aggr_func=aggr_func, activation=None, heads=heads[-1], residual=residual,
                           feat_drop=feat_drop, attn_drop=attn_drop, leaky_relu_alpha=leaky_relu_alpha))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class HeteroMultiHeadGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, ntypes, etypes, type2mask, num_heads,
                 activation=None, aggr_func="attention", in_dropout=0.1, hidden_dropout=0.1,
                 output_dropout=0.0):
        super(HeteroMultiHeadGCN, self).__init__()
        self.multi_features = isinstance(in_dim, list)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            HeteroMultiHeadGCNLayer(in_feats=in_dim, out_feats=hidden_dim, ntypes=ntypes, etypes=etypes,
                                    type2mask=type2mask, num_heads=num_heads, merge="mean",
                                    aggr_func=aggr_func, activation=activation, dropout_prob=in_dropout))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(
                HeteroMultiHeadGCNLayer(in_feats=hidden_dim, out_feats=hidden_dim, ntypes=ntypes, etypes=etypes,
                                        type2mask=type2mask, num_heads=num_heads, merge="mean",
                                        aggr_func=aggr_func, activation=activation, dropout_prob=hidden_dropout))
        # output layer
        self.layers.append(
            HeteroMultiHeadGCNLayer(in_feats=hidden_dim, out_feats=out_dim, ntypes=ntypes, etypes=etypes,
                                    type2mask=type2mask, num_heads=num_heads, merge="mean",
                                    aggr_func=aggr_func, activation=None, dropout_prob=output_dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
