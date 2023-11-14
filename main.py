import os
import pandas as pd
import numpy as np
import random
from pprint import pprint
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR,CosineAnnealingWarmRestarts
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback

# transformer:
from transformers import BertTokenizer, AdamW, BertModel, RobertaTokenizer,RobertaModel #XLNetTokenizer,XLNetModel,AutoFeatureExtractor

# custom
from loss import loss_function, true_metric_loss
import configparser
        
# mult
from modules.mulT_modules.transformerEncoder import TransformerEncoder
from modules.mulT_modules.mulT import MULTModel

## dgl
from modules.gnn_modules.build_graph import *
from modules.gnn_modules.graphconv import SAGEConv,HeteroGraphConv
from modules.gnn_modules.self_att import Attention

os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND']= 'pytorch'

def th_seed_everything(seed: int = 2023):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
class Arg:
    epochs: int = 50  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 50  # Max Length input size 
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    batch_size: int = 32
    
    
class Model(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        
        # config:
        self.file_path = configparser.ConfigParser()
        self.file_path.read('path.ini')
        
        self.args = args
        self.config = config
        self.seed = self.config['random_seed']
        self.gpu = self.config['gpu']
        self.split = self.config['split']
        self.save = self.config['save']
        self.batch_size = self.args.batch_size
        
        ## setting
        self.chunk_size = self.config['chunk_size']
        self.embed_type = self.config['embed_type']
        
        self.num_labels = self.config['num_labels']
        self.dropout = nn.Dropout(self.config['dropout'])
        self.loss_type = self.config['loss']
        ## col
        self.y_col = self.config['y_col']
        self.modal = self.config['modal']
        self.t, self.a, self.v  = self.modal.split('_') 
        
        self.att = self.config['att']
        self.tonly,self.aonly,self.vonly= self.att.split('_')
        self.output_dim = self.config['num_labels']
        self.txt_col = 'asr_body_pre'
        self.token_num = self.config['num_token']
        if self.embed_type == "bert":
            pretrained = "bert-base-uncased"
            self.tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=True)
            self.model = BertModel.from_pretrained(pretrained)

        elif self.embed_type == "rb":
            pretrained = 'roberta-base'
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained, do_lower_case=True)
            self.model = RobertaModel.from_pretrained(pretrained)            
        
        self.a_hidden = int(self.file_path['hidden'][self.a])
        self.t_hidden = int(self.file_path['hidden'][self.t])
        self.v_hidden = int(self.file_path['hidden'][self.v])
        
        ## dgl
        # Inter Relation define
        self.gnn_size = 768
        self.agg_type = self.config['agg_type']#'lstm'
        self.hetero_type = self.config['hetero_type']
        if self.config['rel_type'] == 'v':
            rel_names = {'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        elif self.config['rel_type'] == 'a':
            rel_names = {'ak': (int(self.a_hidden)*3, int(self.t_hidden)),
                         'ka': (int(self.t_hidden), int(self.a_hidden)*3),
                         'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        elif self.config['rel_type'] == 'va':
            rel_names = {'ak': (int(self.a_hidden)*3, int(self.t_hidden)),
                         'ka': (int(self.t_hidden), int(self.a_hidden)*3),
                         'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        
        # Model init
        mod_dict = {rel : SAGEConv((src_dim, dst_dim), self.gnn_size,
                           aggregator_type = self.agg_type)\
                          for rel,(src_dim, dst_dim) in rel_names.items()}

        self.conv = HeteroGraphConv(mod_dict, aggregate=self.hetero_type)
        
        self.v_lstm = nn.LSTM(input_size=self.v_hidden,
                             hidden_size=self.v_hidden,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True)
        
        self.a_lstm = nn.LSTM(input_size=self.a_hidden,
                             hidden_size=self.a_hidden,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True)
        
        self.v_atten = Attention(self.config['gpu'],int(self.v_hidden *2), batch_first=True)  # 2 is bidrectional
        self.a_atten = Attention(self.config['gpu'],int(self.a_hidden *2), batch_first=True)  # 2 is bidrectional
        self.dropout = nn.Dropout(self.config['dropout'])

        # model & hidden
        if self.config['graphuse']:
            self.MULTModel = MULTModel(self.config,self.file_path,use_origin=True)
        else:
            self.MULTModel = MULTModel(self.config,self.file_path,use_origin=False)
            
        self.fc1 = nn.Linear(self.t_hidden, int(self.t_hidden/2))
        self.fc2 = nn.Linear(int(self.t_hidden/2), self.output_dim)
        
    def forward(self,labels,txt,aud,vid,idx_, **kwargs):
        # 0. Input
        batch_size, seq_ =txt.shape
        v_h,_ = self.v_lstm(vid)
        v_h, v_att_score = self.v_atten(v_h)
        
        a_h,_ = self.a_lstm(aud)
        a_h, a_att_score = self.a_atten(a_h)
        

        def historic_feat(feat):
            next_ = torch.cat([feat[1:,:,:], feat[0,:,:].unsqueeze(0)],axis=0)
            past_ = torch.cat([feat[-1,:,:].unsqueeze(0),feat[:-1,:,:]],axis=0)

            feat =torch.cat([feat,next_,past_],axis=2)
            return feat
        vid_h = historic_feat(v_h)
        aud_h = historic_feat(a_h)
        
        
        # 1. Speech-gesture graph encoder
        if self.config['graphuse']:
            bg = dgl.batch([self.g_list[idx] for idx in idx_])#.to(device)    
            bg.ndata['features']['a'] = a_h.reshape(-1,int(self.a_hidden*2))
            bg.ndata['features']['v'] = v_h.reshape(-1,int(self.v_hidden*2))
            bg.ndata['features']['k'] = self.key_embed.repeat(batch_size,1,1).reshape(-1,self.t_hidden)
            
            if self.config['edge_weight']:
                mod_args = bg.edata['weights'] #{'edge_weight': bg.edata['weights']}
            else:
                mod_args = None
            gnn_h = self.conv(g=bg, inputs=bg.ndata['features'], mod_args=mod_args)
        
        
        # 2. Gesture-aware embedding update
        if self.config['update']:
            key_embed = gnn_h['k'].reshape(batch_size,-1,self.gnn_size) # 32 x 30 x 20

            with torch.no_grad():
                new_embedding = self.model.embeddings.word_embeddings.weight.data.clone()
                new_embedding[self.keyword_token] = key_embed[0] 
                self.model.embeddings.word_embeddings.weight.set_(new_embedding)

        if self.config['graphuse']:
            if self.config['rel_type'] == 'v':
                aud_h = gnn_h['v'].reshape(batch_size,-1,self.gnn_size) # 32 x 30 x 20 [batch*node, hidden_dim]
                vid_h = aud_h
            elif self.config['rel_type'] == 'a':
                vid_h = gnn_h['a'].reshape(batch_size,-1,self.gnn_size) # 32 x 30 x 20
                aud_h = vid_h 
                
            elif self.config['rel_type'] == 'va':
                aud_h = gnn_h['a'].reshape(batch_size,-1,self.gnn_size) # 32 x 30 x 20 [batch*node, hidden_dim]
                vid_h = gnn_h['v'].reshape(batch_size,-1,self.gnn_size) # 32 x 30 x 20
        txt_h = self.model(input_ids =txt)
        
        # 3. Multimodal Fusion Encoder
        relation_h,_, att_vl =self.MULTModel(txt_h[0], aud_h, vid_h) # 32 x 20
        last_h_l = txt_h[1]+relation_h
        
        # 4. Aphasia Type Detection
        logits = self.fc2(F.relu(self.fc1(last_h_l)))
        loss = loss_function(logits, labels, self.loss_type, self.num_labels, 1.8)

        return logits,loss,att_vl,v_att_score

    def configure_optimizers(self):
        if self.config['optimizer'] == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        elif self.config['optimizer'] == 'adamwp':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.config['lr'])
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
            
        if self.config['lr_scheduler'] == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.config['lr_scheduler'] == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.2)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
            
            
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess_dataframe(self):
        # Data Load
        data_path = self.file_path['data_path']['chunk' + str(self.chunk_size)]    
        df = pd.read_json(data_path)           

        aud_feat = np.load(self.file_path['feature_path'][self.a + str(self.chunk_size + 2)]) # +2 : BERT CLS, EOS token num
        vid_feat = np.load(self.file_path['feature_path'][self.v+ str(self.chunk_size + 2)])   
        
        # Add Disfluency Tokens
        with open(f'./dataset/disfluency_tk_300.json','r') as f:
            keywords = json.load(f)
    
        keywords = keywords[:self.token_num]
        if self.config['update']:
            print("vocab size (before) : ", len(self.tokenizer))
            self.tokenizer.add_tokens(keywords, special_tokens=False) 
            print("vocab size (after) : ", len(self.tokenizer))
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        self.keyword_token = torch.tensor(self.tokenizer.encode(' '.join(keywords))[1:-1])
        self.key_embed = self.model(input_ids =self.keyword_token.unsqueeze(0))[0] # 200X768

        # Tokenizer
        print('tokenizing')
        df[self.txt_col] = df[self.txt_col].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        
        # Heterogeneous Graph Construction 
        if self.config['graphuse']:
            print('build graph')
            self.g_list = build_graph(self.config,self.file_path).data_load(self.gpu)
        
        
        # Stratified GroupKfold
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
        for i,(train_idxs, test_idxs) in enumerate(kf.split(df, df['type_label'], df['user_name'])):
            if i == self.split:
                break
            pprint(f"data Size: {len(train_idxs)}, {len(test_idxs)}")
                
        df_train = df.iloc[train_idxs]
        df_test = df.iloc[test_idxs]
              
        # Gender Setting
        if self.config['train_gender'] == 'female':
            df_train = df_train[df_train.sex == 'female']

        elif self.config['train_gender'] == 'male':
            df_train = df_train[df_train.sex == 'male']
            
        train_idxs = df_train.index.tolist()
        pprint(f"data Size: {len(df_train)}, {len(df_test)}")

        
        # Dataloader
        self.train_data = TensorDataset(
            torch.tensor(df_train['data_id'].tolist(), dtype=torch.long),
            torch.tensor(df_train[self.y_col].tolist(), dtype=torch.long),
            torch.tensor(df_train[self.txt_col].tolist(), dtype=torch.long),
            torch.tensor(np.nan_to_num(aud_feat[train_idxs].astype('float64')), dtype=torch.float),
            torch.tensor(np.nan_to_num(vid_feat[train_idxs].astype('float64')), dtype=torch.float),
            torch.tensor(train_idxs, dtype=torch.long)
        )
        
        self.test_data = TensorDataset(
            torch.tensor(df_test['data_id'].tolist(), dtype=torch.long),
            torch.tensor(df_test[self.y_col].tolist(), dtype=torch.long),
            torch.tensor(df_test[self.txt_col].tolist(), dtype=torch.long),
            torch.tensor(np.nan_to_num(aud_feat[test_idxs].astype('float64')), dtype=torch.float),
            torch.tensor(np.nan_to_num(vid_feat[test_idxs].astype('float64')), dtype=torch.float),
            torch.tensor(test_idxs, dtype=torch.long)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        data_id, labels, txt,aud, vid,idx_ = batch 
        logits,loss,att_vl,v_att_score = self(labels, txt,aud, vid,idx_) 
        self.log("train_loss", loss)   
        
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        data_id, labels, txt,aud, vid,idx_ = batch 
        logits,loss,att_vl,v_att_score = self(labels, txt,aud, vid,idx_) 

        att_save = list(att_vl.detach().cpu().numpy())
        lstm_att_save = list(v_att_score.detach().cpu().numpy())
        
        preds = logits.argmax(dim=-1)
        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())
        data_id = list(data_id.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
            'data_id': data_id,
            'att_save': att_save,
            'lstm_att_save': lstm_att_save,
        }

    def test_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        y_true = []
        y_pred = []
        data_id = []

        for i in outputs:
            loss += i['loss'].cpu().detach()
            y_true += i['y_true']
            y_pred += i['y_pred']
            data_id += i['data_id']
        
        _loss = loss / len(outputs)
        loss = float(_loss)
            
        y_true = np.asanyarray(y_true)
        y_pred = np.asanyarray(y_pred)
        data_id = np.asanyarray(data_id)
            
        val_score = classification_report(y_true, y_pred, output_dict=True) 
        print(val_score)
        
        val_df = pd.DataFrame.from_dict(val_score).T.reset_index()
        val_df = val_df.rename(columns = {'index':'category'})
        val_df['save'] = self.save
        val_df['chunk_size'] = self.chunk_size
        val_df['test_size'] = len(y_pred)
        val_df['split'] = self.split
        val_df['y_col'] = self.y_col
        val_df['modal'] = self.modal
        val_df['embed'] = self.embed_type
        val_df['agg_type'] = self.agg_type
        val_df['hetero_type'] = self.agg_type
        val_df['config'] = str(self.config)
        
        pred_dict = { 
            'save' : [self.save]*len(y_pred),
            'data_id' : data_id,
            'chunk_size': [self.chunk_size]*len(y_pred),
            'test_size':  [len(y_pred)]*len(y_pred),
            'split': [self.split]*len(y_pred),
            'y_col': [self.y_col]*len(y_pred),
            'modal': [self.modal]*len(y_pred),
            'embed': [self.embed_type]*len(y_pred),
            'agg_type':[self.agg_type]*len(y_pred),
            'hetero_type':[self.hetero_type]*len(y_pred),
            'config':[str(self.config)]*len(y_pred),
            
            'true' : y_true,
            'pred' : y_pred,
        }
        
        # Result Save
        self.save_path = f"./_result/"
        Path(f"{self.save_path}/pred").mkdir(parents=True, exist_ok=True)
        self.save_time = datetime.now().__format__("%m%d_%H%M%S%Z")
        pd.DataFrame(val_df).to_csv(f'{self.save_path}{self.save_time}_{self.save}.csv',index=False)  
        pd.DataFrame(pred_dict).to_csv(f'{self.save_path}pred/{self.save_time}_{self.save}_pred.csv',index=False)

    
def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything(config['random_seed'])
    th_seed_everything(config['random_seed'])
    
    model = Model(args,config) 
    model.preprocess_dataframe()

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    print(":: Start Training ::")
    trainer = Trainer(
        logger = False,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        callbacks=[early_stop_callback],
        deterministic=False, # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        # For GPU Setup
        accelerator='gpu',
        devices=[config['gpu']]if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32
    )
    
    trainer.fit(model,train_dataloaders = model.train_dataloader())
    trainer.test(model,dataloaders=model.test_dataloader())

if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--random_seed", type=int, default=2023) 
    parser.add_argument("--dropout", type=float, default=0.01,help="dropout probablity")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--gpu", type=int, default=1,  help="save fname")
    parser.add_argument("--optimizer", type=str, default='adamw')                
    parser.add_argument("--lr_scheduler", type=str, default='exp')   
    parser.add_argument("--loss", type=str, default="cross") 
    parser.add_argument("--split", type=int, default=0) # 0~5 
    parser.add_argument("--chunk_size", type=int, default=50) 
    parser.add_argument("--y_col", type=str, default='type_label')  #label fre_label com_label
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--modal", type=str, default="t_a_v") # a, am, t, v
    parser.add_argument("--att", type=str, default="t_a_v") # a, am, t, v
    parser.add_argument("--embed_type", type=str, default="rb") 
    parser.add_argument("--agg_type", type=str, default="bilstm") 
    parser.add_argument("--hetero_type", type=str, default="min") 
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--no-update', dest='update', action='store_false')
    parser.add_argument('--edge_weight', action='store_true')
    parser.add_argument('--no-edge_weight', dest='edge_weight', action='store_false')
    parser.add_argument('--graphuse', action='store_true')
    parser.add_argument('--no-graphuse', dest='graphuse', action='store_false')
    parser.add_argument('--rel_type', type=str, default='va')
    parser.add_argument('--train_gender', type=str, default='both')
    parser.add_argument("--num_token", type=int, default=150) 
    parser.add_argument("--save", type=str, default="rebuttal") 
    
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__) 


