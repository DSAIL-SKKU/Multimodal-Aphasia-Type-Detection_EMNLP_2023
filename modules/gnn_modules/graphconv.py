import os
import pandas as pd
import numpy as np
from pprint import pprint
# torch:
import torch
from torch import nn
import torch.nn.functional as F

#gcn
from scipy.sparse import coo_matrix
import dgl
import dgl.nn.pytorch as dglnn
from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape
from dgl.base import DGLError
from functools import partial
import math
from tqdm import tqdm            

    
class SAGEConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
        #activation=F.relu,
        kernel_out = 20,
        device = 1
    
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm","bilstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.gpu = device
        
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )
            
        if aggregator_type == "bilstm":
            self.bilstm = nn.LSTM(
                input_size=self._in_src_feats,
                hidden_size=self._in_src_feats,
                bidirectional=True,
                batch_first=True) #                   num_layers=2,
            self.fc_rst = nn.Linear(self._in_src_feats*2, self._in_src_feats)

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type == "bilstm":
            self.bilstm.reset_parameters()
            #self.atten.reset_parameters()
            
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

            
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)


    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        
        return {"neigh": rst.squeeze(0)}
    
    def _bilstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((2, batch_size, self._in_src_feats)),
            m.new_zeros((2, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.bilstm(m,h)#, h
        #print(rst.shape)
        rst = torch.cat([rst[0, :, :], rst[1, :, :]], dim=1).unsqueeze(0)
        #print(rst.shape)
        
        rst = self.fc_rst(rst)
        #print(rst.shape)
        
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):

        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                            : graph.num_dst_nodes()
                        ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "bilstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._bilstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
                
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst

#################################################    
    
class HeteroGraphConv(nn.Module):

    def __init__(self, mods, aggregate="sum"):
        super(HeteroGraphConv, self).__init__()
        self.mod_dict = mods
        mods = {str(k): v for k, v in mods.items()}
        # Register as child modules
        self.mods = nn.ModuleDict(mods)
        # PyTorch ModuleDict doesn't have get() method, so I have to store two
        # dictionaries so that I can index with both canonical edge type and
        # edge type with the get() method.
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(
                v, "set_allow_zero_in_degree", None
            )
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate
        
        self.depth = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2),
            nn.Conv2d(2, 1, kernel_size=1)
        ) 


    def _get_module(self, etype):
        mod = self.mod_dict.get(etype, None)
        if mod is not None:
            return mod
        if isinstance(etype, tuple):
            # etype is canonical
            _, etype, _ = etype
            return self.mod_dict[etype]
        raise KeyError("Cannot find module with edge type %s" % etype)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    graph=rel_graph,
                    feat=(src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get((stype, etype, dtype), ()),
                    **mod_kwargs.get((stype, etype, dtype), {})
                )
                outputs[dtype].append(dstdata)
                
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    graph=rel_graph,
                    feat=(inputs[stype], inputs[dtype]),
                    *mod_args.get((stype, etype, dtype), ()),
                    **mod_kwargs.get((stype, etype, dtype), {})
                )
                
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty, self.depth)
        return rsts

def _max_reduce_func(inputs, dim):
    return torch.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return torch.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return torch.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return torch.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype):  # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return torch.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, models,fn):  # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = torch.stack(inputs, dim=0)
    return fn(stacked, dim=0)

def _depth_agg_func(inputs, dsttype, models):
    
    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs[0]
    
    out = models(torch.stack(inputs, dim=0).unsqueeze(0))
    
    return out.squeeze(0).squeeze(0)#.double()



def get_aggregate_fn(agg):

    if agg == "sum":
        fn = _sum_reduce_func
    elif agg == "max":
        fn = _max_reduce_func
    elif agg == "min":
        fn = _min_reduce_func
    elif agg == "mean":
        fn = _mean_reduce_func
    elif agg == "stack":
        fn = None  # will not be called
    elif agg == 'depth':
        fn = None  # will not be called
    else:
        raise DGLError(
            "Invalid cross type aggregator. Must be one of "
            '"sum", "max", "min", "mean" ,"depth" or "stack". But got "%s"' % agg
        )
    if agg == "stack":
        return _stack_agg_func
    elif agg == 'depth':
        return _depth_agg_func
    else:
        return partial(_agg_func, fn=fn)
    

    
    
                