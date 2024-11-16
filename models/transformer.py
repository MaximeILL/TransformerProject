import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # weights for linear projections of inputs
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calculates the scaled dot-product attention
        Q (Tensor): Query tensor of shape (..., seq_len_q, d_k)
        K (Tensor): Key tensor of shape (..., seq_len_k, d_k)
        V (Tensor): Value tensor of shape (..., seq_len_v, d_v)
        mask (tensor, optional) : mask tensor to control which elements are considered in the attention scores
        returns tensor : output tensor resulting from attention applied to the input values
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)

        # Store attention weights for visualization
        self.attn_probs = attn_probs.detach()

        return output

    def split_heads(self, x):
        """
        Splits input tensor into multiple heads for multi-head attention
        x (Tensor): Shape (batch_size, seq_length, d_model)
        returns tensor: reshaped to (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combines multi-head tensors into a single tensor
        x (Tensor): Shape (batch_size, num_heads, seq_length, d_k)
        returns tensor : reshaped to (batch_size, seq_length, d_model)
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Performs forward pass for multi-head attention
        Q (Tensor): Query tensor
        K (Tensor): Key tensor
        V (Tensor): Value tensor
        mask (tensor, optional) : Masking tensor to prevent attention on certain positions
        returns tensor: Output tensor after applying multi-head attention
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        # compute attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.dropout(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)

        self.attn_probs = self.self_attn.attn_probs

        x = self.norm1(x + self.dropout(attn_output)) 
        ff_output = self.feed_forward(x) 
        x = self.norm2(x + self.dropout(ff_output)) 
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)

        # Store decoder self-attention weights
        self.self_attn_probs = self.self_attn.attn_probs

        x = self.norm1(x + self.dropout(attn_output))  
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)

        # Store cross-attention weights
        self.cross_attn_probs = self.cross_attn.attn_probs

        x = self.norm2(x + self.dropout(attn_output)) 
        ff_output = self.feed_forward(x) 
        x = self.norm3(x + self.dropout(ff_output)) 
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Generates masks for the source and target sequences
        src (Tensor) : Source tensor where non-zero values indicate valid tokens
        tgt (Tensor) : Target tensor where non-zero values indicate valid tokens
        returns Tuple[Tensor, Tensor] : Source mask and target mask tensors
        
        this function creates a mask for the source sequence by identifying non-zero tokens
        For the target sequence, creates a mask that combines non-zero tokens and 
        a no-peak mask to prevent attending to future positions
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = torch.triu(torch.ones((1, seq_length, seq_length), dtype=torch.bool), diagonal=1)
        tgt_mask = tgt_mask & ~nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        self.encoder_attn_weights = []  

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            self.encoder_attn_weights.append(enc_layer.attn_probs)

        dec_output = tgt_embedded
        self.decoder_self_attn_weights = [] 
        self.decoder_cross_attn_weights = []  

        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            self.decoder_self_attn_weights.append(dec_layer.self_attn_probs)
            self.decoder_cross_attn_weights.append(dec_layer.cross_attn_probs)

        output = self.fc(dec_output)
        return output
