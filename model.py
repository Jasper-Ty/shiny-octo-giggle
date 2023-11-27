import math
from torch import Tensor
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Additive encoding for position according to the "Attention is all you need" paper
    """
    def __init__(self, d_emb=4, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # initialize encoding tensor (max_len X d_emb)
        pe = torch.zeros(max_len, d_emb)

        # creates a (1 X max_len) tensor, which looks like
        # [ 
        #   [0]
        #   [1]
        #   [2]
        #    : 
        #   [max_len]
        # ]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Computes a tensor (d_emb/2) consisting of terms (10000^(2k/d_emb))
        div_term = torch.exp(torch.arange(0, d_emb, 2, dtype=torch.float) * (-math.log(10000.0) / d_emb))

        # multiplication in pytorch follows numpy broadcasting semantics
        # see https://numpy.org/doc/stable/user/basics.broadcasting.html
        # so position * div_term is actually a (maxlen X d_emb/2) tensor
        # and torch.sin performs elementwise
        # then use clever slicing to fill in odd and even elements
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        # register_buffer() ensures pe is NOT considered a model parameter
        # i.e is not trained at all
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:x.size(0), :]
        return x



class Model(nn.Module):
    def __init__(self, d_emb=4, perm_length=7):
        super(Model, self).__init__()

        self.transformer = nn.Transformer(
            d_model=d_emb, 
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            batch_first=True,
        )

        self.generator = nn.Linear(d_emb, perm_length)
        self.src_emb = nn.Embedding(perm_length, d_emb)
        self.tgt_emb = nn.Embedding(perm_length, d_emb)
        self.positional_encoding = PositionalEncoding(d_emb)

    def forward(self, 
                src: Tensor, 
                tgt: Tensor, 
                src_mask: Tensor, 
                tgt_mask: Tensor
                ):
        src_emb = self.positional_encoding(self.src_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        src_emb = self.positional_encoding(self.src_emb(src))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt_emb = self.positional_encoding(self.tgt_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask():
    src_seq_len = 10
    tgt_seq_len = 7 

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    return src_mask, tgt_mask, 
