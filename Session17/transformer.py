import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F

import math
from dataclasses import dataclass


@dataclass
class Config:
    name : str = "Bert"
    n_code: int = 8
    n_heads: int = 8
    embed_size: int = 128
    inner_ff_size: int = embed_size * 4
    n_embeddings: int = 768
    seq_len:int  = 20
    dropout: float = 0.1

def encoder_block(config):
    return  nn.TransformerEncoderLayer(d_model=config.embed_size, 
                                        nhead=config.n_heads,
                                        dim_feedforward=config.inner_ff_size,
                                        dropout=config.dropout,
                                        activation="relu", 
                                        batch_first=True, # Do our batches come first?
                                        norm_first=True) # Normalize first or after MSA/MLP layers?

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len



                                            
class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embeddings is not None
        assert config.seq_len is not None
        self.config = config
        
        self.model_dict = {
            'BERT': nn.ModuleDict({
                'wte': nn.Embedding(config.n_embeddings, config.embed_size),
                'wpe': PositionalEmbedding(config.embed_size, config.seq_len),
                'h': nn.ModuleList([encoder_block(config) for _ in range(config.n_code)]),
                'ln_f': nn.LayerNorm(config.embed_size)
            }),

            'GPT': nn.ModuleDict({
                'wte': nn.Embedding(config.n_embeddings, config.embed_size),
                'wpe': nn.Embedding(config.seq_len, config.embed_size),
                'h': nn.Sequential(*[encoder_block(config) for _ in range(config.n_code)]),
                'ln_f': nn.LayerNorm(config.embed_size)
            })
        }
        self.transformer = self.model_dict.get(config.name,None)
        assert self.transformer is not None
        if config.name == "GPT":
             self.register_buffer("tril", torch.tril(torch.ones(config.seq_len, config.seq_len)))
            
        self.lm_head = nn.Linear(config.embed_size,config.n_embeddings, bias=False)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if hasattr(self.transformer.wpe,'weight'):
                n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            self.eval()
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def forward(self,x,targets=None):
        device = x.device
        b, t = x.size()
        assert t <= self.config.seq_len, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        x = self.transformer.wte(x)
        if isinstance(self.transformer.wpe,nn.Embedding):
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos)
        else:
            pos_emb = self.transformer.wpe(x)
        
        x = x + pos_emb
        if self.config.name == 'GPT':
            for block in self.transformer.h:
                if self.training:
                    x = block(x,src_mask=self.tril)
                else:
                    x = block(x)
        else:
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets != None:
            # cross_entropy accepts inputs in a (batch_size, num_classes)
            # so we need to reformat our logits dimensions to
            # (batch_size * time, dim_vocabulary), time = block_size
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            targets = torch.reshape(targets, (B * T,))
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
      
 # 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        self.patch_size = patch_size
        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method 
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        # 6. Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]       

@dataclass
class Vit_Config:
    img_size:int=224 
    in_channels:int=3
    patch_size:int=16 
    num_transformer_layers:int=12
    embed_size:int=768 
    inner_ff_size:int=3072 
    n_heads:int=12 
    attn_dropout:float=0 
    dropout:float=0.1  
    embedding_dropout:float=0.1
    num_classes:int=1000 
    activation:str = 'gelu'



        


class Vision_Transformer(nn.Module):
    def __init__(self, vit_config):
        super().__init__()
        vit_config.activation = 'gelu'
        self.num_patches = (vit_config.img_size * vit_config.img_size) // (vit_config.patch_size**2)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, vit_config.embed_size),requires_grad=True)
        self.position_embedding =  nn.Parameter(data=torch.randn(1, self.num_patches + 1,vit_config.embed_size), requires_grad=True)
        
        self.vit_transformer = nn.ModuleDict({
            'embedding_dropout': nn.Dropout(p=vit_config.embedding_dropout),
            'patch_embedding': PatchEmbedding(in_channels=vit_config.in_channels, patch_size=vit_config.patch_size, embedding_dim=vit_config.embed_size),
            'transformer_encoder': nn.Sequential(*[encoder_block(vit_config) for _ in range(vit_config.num_transformer_layers)]),
            'classifier': nn.Sequential(nn.LayerNorm(normalized_shape=vit_config.embed_size),
                                        nn.Linear(in_features=vit_config.embed_size, out_features=vit_config.num_classes))
        })

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.vit_transformer['patch_embedding'](x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.vit_transformer['embedding_dropout'](x)
        x = self.vit_transformer['transformer_encoder'](x)
        x = self.vit_transformer['classifier'](x[:, 0])  # Take the output at the 0 index for classification
        return x













