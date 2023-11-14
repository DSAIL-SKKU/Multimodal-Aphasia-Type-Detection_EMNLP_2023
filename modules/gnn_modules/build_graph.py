import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import itertools
import os
os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
from sklearn.preprocessing import MinMaxScaler
import configparser


class build_graph:

    def __init__(self,config,file_path):

        chunk_size = config['chunk_size']
        num_token = config['num_token']
        
        self.config = config
        def historic_feat(feat):
            next_ = np.append(feat[1:,:,:], np.expand_dims(feat[0,:,:],axis=0),axis=0)#.shape
            past_ = np.append(np.expand_dims(feat[-1,:,:],axis=0), feat[:-1,:,:], axis=0)#.shape

            historic = np.concatenate([feat,next_,past_],axis=2)
            
            return historic
        
        
        # adj matrix
        self.u_w = np.load(file_path['graph']['adj'+ str(chunk_size)])[:,:,:num_token]
        self.u_w = MinMaxScaler().fit_transform(self.u_w.reshape(self.u_w.shape[0], -1)).reshape(self.u_w.shape)
        
        
        # audio, video feature 
        self.a_feat = np.load(file_path['feature_path']['a' + str(chunk_size)])
        self.v_feat = np.load(file_path['feature_path']['v' + str(chunk_size)])
        
        self.a_feat = historic_feat(self.a_feat)
        self.v_feat = historic_feat(self.v_feat)
        
        self.k_feat = np.load(file_path['feature_path']['k'])[:num_token,:]
        
        
        print(self.u_w.shape, self.a_feat.shape, self.v_feat.shape, self.k_feat.shape)
        
    def data_load(self,device):
        graph_list = []
        
        for i, arr in tqdm(enumerate(self.u_w)):
            u_w_mat = torch.tensor(arr.nonzero()).to(f"cuda:{device}")

            #feat
            a_feat = torch.tensor(np.nan_to_num(self.a_feat[i]),dtype=torch.float).to(f"cuda:{device}")
            v_feat = torch.tensor(np.nan_to_num(self.v_feat[i]),dtype=torch.float).to(f"cuda:{device}")
            k_feat = torch.tensor(np.nan_to_num(self.k_feat),dtype=torch.float).to(f"cuda:{device}")       
            e_feat = torch.tensor([arr[u,v] for u,v in zip(u_w_mat[0], u_w_mat[1])],dtype=torch.float).to(f"cuda:{device}")
            # base node - target user
            data_dict = {} #edge
            num_nodes_dict = {} #num_node
            node_feat_dict = {} #node_feat
            edge_feat_dict = {} 
            
            if self.config['rel_type'] == 'v':
                data_dict[('v', 'vk', 'k')] = (u_w_mat[0],u_w_mat[1])
                data_dict[('k', 'kv', 'v')] = (u_w_mat[1],u_w_mat[0])
                num_nodes_dict['v'] = arr.shape[0]
                node_feat_dict['v'] = v_feat
                edge_feat_dict['vk'] = e_feat
                edge_feat_dict['kv'] = e_feat
            
            elif self.config['rel_type'] == 'a':
                data_dict[('a', 'ak', 'k')] = (u_w_mat[0],u_w_mat[1])
                data_dict[('k', 'ka', 'a')] = (u_w_mat[1],u_w_mat[0])
                num_nodes_dict['a'] = arr.shape[0]
                node_feat_dict['a'] = a_feat
                edge_feat_dict['ak'] = e_feat
                edge_feat_dict['ka'] = e_feat
                
            elif self.config['rel_type'] == 'va':
                data_dict[('v', 'vk', 'k')] = (u_w_mat[0],u_w_mat[1])
                data_dict[('k', 'kv', 'v')] = (u_w_mat[1],u_w_mat[0])
                num_nodes_dict['v'] = arr.shape[0]
                node_feat_dict['v'] = v_feat
                edge_feat_dict[('v', 'vk', 'k')] = e_feat
                edge_feat_dict[('k', 'kv', 'v')] = e_feat

                data_dict[('a', 'ak', 'k')] = (u_w_mat[0],u_w_mat[1])
                data_dict[('k', 'ka', 'a')] = (u_w_mat[1],u_w_mat[0])
                num_nodes_dict['a'] = arr.shape[0]
                node_feat_dict['a'] = a_feat
                edge_feat_dict[('a', 'ak', 'k')] = e_feat
                edge_feat_dict[('k', 'ka', 'a')] = e_feat
            
            
            num_nodes_dict['k'] = arr.shape[1]
            node_feat_dict['k'] = k_feat
            
            #graph
            g = dgl.heterograph(data_dict = data_dict,num_nodes_dict = num_nodes_dict).to(f"cuda:{device}")
            g.ndata['features'] = node_feat_dict
            
            if self.config['edge_weight']:
                g.edata['weights'] = edge_feat_dict
                
            graph_list.append(g)

        return graph_list