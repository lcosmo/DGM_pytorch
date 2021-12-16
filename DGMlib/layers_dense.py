import torch
import numpy

from torch.nn import Module, ModuleList, Sequential
from torch import nn

#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
#     x_norm = (x**2).sum(dim,keepdim=True)
#     dist = 2.0 * x_norm - 2.0 * torch.bmm(x, torch.transpose(x, -1, -2))
    x = x - x.mean(-2,keepdim=True)
#     print(x.mean(-2,keepdim=True).shape)
    dist = torch.cdist(x,x)**2
    return dist, x

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1 
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)
    
    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x

def sparse_eye(size):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).float().expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size])) 

class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(DGM_d, self).__init__()
        
        self.sparse=sparse
        
        self.temperature = nn.Parameter(torch.tensor(0).float())
#         self.threshold = nn.Parameter(torch.tensor(0.2).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.k = k
#         self.distance = distance
        
        self.debug=False
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances
        
    def forward(self, x, A, not_used=None, fixedges=None):

        if self.training:
            x = self.embed_f(x,A)  
            if fixedges is not None:                
                return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
            
        
            D, _x = self.distance(x)

            #remove self loop:
            torch.diagonal(D,0,1,2).add_(1e20)

            #sampling here
            edges_hat, logprobs = self.sample_without_replacement(-D)
                
        else:
            with torch.no_grad():
                x = self.embed_f(x,A)  
                if fixedges is not None:                
                    return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
                D, _x = self.distance(x)

                #remove self loop:
                torch.diagonal(D,0,1,2).add_(1e20)
    
                #sampling here
                edges_hat, logprobs = self.sample_without_replacement(-D)

              
        if self.debug:
            self.D = D
            self.edges_hat=edges_hat
            self.logprobs=logprobs
            self.x=x

        return x, edges_hat, logprobs
    

    def sample_without_replacement(self, logits):
        b,n,_ = logits.shape
#         logits = logits * torch.exp(self.temperature*10)
        logits = logits * torch.exp(torch.clamp(self.temperature,-4,4))
        P = torch.softmax(logits+1e-6,-1)
        
        #use topk
        lq = logits
#         if self.training:
        q = torch.rand_like(logits) + 1e-8
        lq = (logits-torch.log(-torch.log(q)))
        logprobs, indices = torch.topk(lq,self.k)  
    
#         logprobs = torch.gather(P, -1, indices)
#         logprobs = torch.gather(logits, -1, indices)
        
        rows = torch.arange(n).view(1,n,1).to(logits.device).repeat(b,1,self.k)
#         edges = torch.stack((rows.view(b,-1),indices.view(b,-1)),-2)
        edges = torch.stack((indices.view(b,-1),rows.view(b,-1)),-2)

        if self.sparse:
            return (edges+(torch.arange(b).to(logits.device)*n)[:,None,None]).transpose(0,1).reshape(2,-1), logprobs
        return edges, logprobs
    
class DGM_c(nn.Module):
    def __init__(self, embed_f, distance=pairwise_euclidean_distances):
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.distance = distance
        
    def forward(self, x, A):
        
        x = self.embed_f(x,A)  
        if self.centroid is None:            
            self.centroid = x.mean(-2,keepdim=True).detach()
            self.scale = (0.9/(x-self.centroid).abs().max()).detach()
        D, x = self.distance( (x-self.centroid)*self.scale)
        self.x = x
        A = torch.sigmoid(self.temperature*(self.threshold.abs()-D))
        self.A=A
#         A = A/A.sum(-1,keepdim=True)
        return x, A
 

class MLP(nn.Module):
    def __init__(self, layers_size,final_activation=False):
        super(MLP, self).__init__()
        layers = []
        for li in range(1,len(layers_size)):
            layers.append(nn.Linear(layers_size[li-1],layers_size[li]))
            if li==len(layers_size)-1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))
            
        self.MLP = nn.Sequential(*layers)
        
    def forward(self, x, e=None):
        x = self.MLP(x)
        return x
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, *params):
        return params