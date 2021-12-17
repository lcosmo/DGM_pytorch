import os
import torch
import numpy as np

import torch_geometric
from torch import nn
from torch.nn import Module, ModuleList, Sequential
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv
from torch.utils.data import DataLoader
import torch_scatter

import pytorch_lightning as pl
from argparse import Namespace

from DGMlib.layers import *
if (not os.environ.get("USE_KEOPS")) or os.environ.get("USE_KEOPS")=="False":
    from DGMlib.layers_dense import *
    
class DGM_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(DGM_Model,self).__init__()
        
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
        
#         self.hparams=hparams
        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        k = hparams.k

            
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 
        for i,(dgm_l,conv_l) in enumerate(zip(dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                if 'ffun' not in hparams or hparams.ffun == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0),k=hparams.k,distance=hparams.distance))
#                 self.graph_f.append(DGM_d(GCNConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
            else:
                self.graph_f.append(Identity())
            
            if hparams.gfun == 'edgeconv':
                conv_l=conv_l.copy()
                conv_l[0]=conv_l[0]*2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            if hparams.gfun == 'gcn':
                self.node_g.append(GCNConv(conv_l[0],conv_l[1]))
            if hparams.gfun == 'gat':
                self.node_g.append(GATConv(conv_l[0],conv_l[1]))
        
        self.fc = MLP(fc_layers, final_activation=False)
        if hparams.pre_fc is not None and len(hparams.pre_fc)>0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None
        
        
        #torch lightning specific
        self.automatic_optimization = False
        self.debug=False
        
    def forward(self,x, edges=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
            x = self.pre_fc(x)
            
        graph_x = x.detach()
        lprobslist = []
        for f,g in zip(self.graph_f, self.node_g):
            graph_x,edges,lprobs = f(graph_x,edges,None)
            b,n,d = x.shape
            
#             edges,_ = torch_geometric.utils.remove_self_loops(edges)
#             edges,_ = torch_geometric.utils.add_self_loops(edges)

            self.edges=edges
            x = torch.nn.functional.relu(g(torch.dropout(x.view(-1,d), self.hparams.dropout, train=self.training), edges)).view(b,n,-1)
            graph_x = torch.cat([graph_x,x.detach()],-1)
            if lprobs is not None:
                lprobslist.append(lprobs)
                
        return self.fc(x),torch.stack(lprobslist,-1) if len(lprobslist)>0 else None
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        
        X, y, mask, edges = train_batch
        edges = edges[0]
        
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        
        pred,logprobs = self(X,edges)
        
        train_pred = pred[:,mask.to(torch.bool),:]
        train_lab = y[:,mask.to(torch.bool),:]
#         train_w = weight[None,mask.to(torch.bool)]    

        #loss = torch.nn.functional.cross_entropy(train_pred.view(-1,train_pred.shape[-1]),train_lab.argmax(-1).flatten())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred,train_lab)
        loss.backward()

        correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()

        #GRAPH LOSS
        if logprobs is not None: 
            corr_pred = (train_pred.argmax(-1)==train_lab.argmax(-1)).float().detach()
            wron_pred = (1-corr_pred)

            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred)*0.5

            point_w = (self.avg_accuracy-corr_pred)#*(1*corr_pred + self.k*(1-corr_pred))
            graph_loss = point_w * logprobs[:,mask.to(torch.bool),:].exp().mean([-1,-2])

            graph_loss = graph_loss.mean()# + self.graph_f[0].Pr.abs().sum()*1e-3
            graph_loss.backward()
            
            self.log('train_graph_loss', graph_loss.detach().cpu())
            if(self.debug):
                self.point_w = point_w.detach().cpu()
                
            self.avg_accuracy = self.avg_accuracy.to(corr_pred.device)*0.95 +  0.05*corr_pred
            
        optimizer.step()

        self.log('train_acc', correct_t)
        self.log('train_loss', loss.detach().cpu())
        
    
    def test_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]
        
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        pred,logprobs = self(X,edges)
        pred = pred.softmax(-1)
        for i in range(1,self.hparams.test_eval):
            pred_,logprobs = self(X,edges)
            pred+=pred_.softmax(-1)
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        self.log('test_loss', loss.detach().cpu())
#         self.log('test_graph_loss', loss.detach().cpu())
        self.log('test_acc', 100*correct_t)
    
    def validation_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]
        
        
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        
        pred,logprobs = self(X,edges)
        pred = pred.softmax(-1)
        for i in range(1,self.hparams.test_eval):
            pred_,logprobs = self(X,edges)
            pred+=pred_.softmax(-1)
        
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        
        self.log('val_loss', loss.detach())
        self.log('val_acc', 100*correct_t)
        
#         ####### visualizations ###########
#         try:
#             self.graph_f[0].debug=True
#             pred,logprobs = self(X)
#             self.graph_f[0].debug=False

#             x = self.graph_f[0].x[0].detach()
#             c = torch.argmax(y,-1)
#             D = self.graph_f[0].distance(x)[0]
#             D.diagonal().fill_(0)

#             sidx = torch.argsort( (c[0]+1)*10 + (mask+1)*1)
#             P = torch.exp(-D[sidx,:][:,sidx]*torch.clamp(self.graph_f[0].temperature.detach().cpu(),-5,5).exp())#>0.001

#             img = PIL.Image.fromarray((P*255).byte().detach().cpu().numpy())
#             img = img.resize((512,512), PIL.Image.ANTIALIAS)

#             I = wandb.Image(img, caption="adj")
#             self.logger.experiment.log({'adj': [I]})
#         except:
#             pass
      

