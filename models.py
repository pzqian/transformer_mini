import torch


class ScaledProductAttention(torch.nn.Module):

    def __init__(self, hidden_size, num_heads=1, d_k=None, d_v=None):
        super(ScaledProductAttention, self).__init__()
        # make sure hidden_size is a tensor
        if not torch.is_tensor(hidden_size):
            hidden_size = torch.tensor(hidden_size)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else hidden_size // num_heads
        self.d_v = d_v if d_v is not None else hidden_size // num_heads
        self.proj_K = torch.nn.Linear(hidden_size, num_heads * self.d_k)
        self.proj_Q = torch.nn.Linear(hidden_size, num_heads * self.d_k)
        self.proj_V = torch.nn.Linear(hidden_size, num_heads * self.d_v)
        self.proj_0 = torch.nn.Linear(num_heads * self.d_v, hidden_size)

    def forward(self, Q, K, V, mask=None):
        """
        Each of Q, K, V is a tensor of shape (batch_size, seq_len, hidden_size). 
            NOTE: seq_len might be different for Q, K, V.
        seq_len can be different for Q, K, V.
        mask is of shape (batch_size, seq_len_q, seq_len_k).
            NOTE: a lot of implementations are assuming seq_len_q and seq_len_k are the same. 
        For example, Q could be the length of the input. K and V could be the length of the output.
        """
        proj_K = self.proj_K(K) # (batch_size, seq_len, num_heads * d_k)
        proj_Q = self.proj_Q(Q) # (batch_size, seq_len, num_heads * d_k)
        proj_V = self.proj_V(V) # (batch_size, seq_len, num_heads * d_v)
        scores = torch.matmul(proj_Q, proj_K.transpose(1, 2)) # (batch_size, seq_len_q, seq_len_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        sm = torch.softmax(scores / torch.sqrt(self.d_k), dim=-1)
        multihead = torch.matmul(sm, proj_V) # (batch_size, seq_len, num_heads * d_v)
        return self.proj_0(multihead) # (batch_size, seq_len, hidden_size)


class PositionWiseFFN(torch.nn.Module):

    def __init__(self, hidden_size, linear_size):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_size, linear_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(linear_size, hidden_size)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


# %%
class LayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
        self.beta = torch.nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# %%
class EncoderBlock(torch.nn.Module):

    def __init__(self, hidden_size):
        super(EncoderBlock, self).__init__()
        self.hidden_size = hidden_size
        self.attention = ScaledProductAttention(hidden_size)
        # Setting the linear size to 4 times the hidden size is a common choice.
        self.feed_forward = PositionWiseFFN(hidden_size, hidden_size * 4)
        self.layer_norm = LayerNorm(hidden_size)
    
    def forward(self, X, mask=None):
        sub_layer_1 = self.layer_norm(self.attention(X, X, X, mask=mask) + X)
        sub_layer_2 = self.layer_norm(self.feed_forward(sub_layer_1) + sub_layer_1)
        return sub_layer_2

# %%
class Encoder(torch.nn.Module):

    def __init__(self, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_blocks = torch.nn.ModuleList([EncoderBlock(hidden_size) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        # concat the output from calling each of self.encoder_blocks on x.
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask=mask)
        return x 


class DecoderBlock(torch.nn.Module):

    def __init__(self, hidden_size):
        super(DecoderBlock, self).__init__()
        self.hidden_size = hidden_size
        self.self_layer_norm = LayerNorm(hidden_size)
        self.self_attention = ScaledProductAttention(hidden_size)
        self.middle_layer_norm = LayerNorm(hidden_size)
        self.middle_attention = ScaledProductAttention(hidden_size)
        self.feed_forward = PositionWiseFFN(hidden_size, hidden_size * 4)
        self.ffn_layer_norm = LayerNorm(hidden_size)

    def forward(self, x, encoder_output, mask=None, memory_mask=None):
        x = self.self_layer_norm(self.self_attention(x, x, x, mask=memory_mask) + x)
        x = self.middle_layer_norm(self.middle_attention(x, encoder_output, encoder_output, mask=mask) + x)
        x = self.ffn_layer_norm(self.feed_forward(x) + x)
        return x



# %%
class Decoder(torch.nn.Module):
    
        def __init__(self, hidden_size, num_layers):
            super(Decoder, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.decoder_blocks = torch.nn.ModuleList([DecoderBlock(hidden_size) for _ in range(num_layers)])
        
        def forward(self, x, encoder_output, mask=None, memory_mask=None):
            for decoder_block in self.decoder_blocks:
                x = decoder_block(x, encoder_output, mask=mask, memory_mask=memory_mask)
            return x


class Transformer(torch.nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.embedding = torch.nn.Embedding(config['vocab_size'], self.hidden_size)
        # TODO: implement sinoid positional embedding as written in the paper.
        self.pos_embedding = torch.nn.Embedding(config['max_seq_len'], self.hidden_size)
        self.max_seq_len = config['max_seq_len'] # This is also called block_size, I think.
        self.padding_idx = config['padding_idx']

        self.encoder = Encoder(self.hidden_size, self.num_layers)
        self.decoder = Decoder(self.hidden_size, self.num_layers)
        # This is actaully different from the embedding matrix.
        self.linear = torch.nn.Linear(self.hidden_size, config['vocab_size'])

    def _gen_max_seq_mask(self, seq_matrix, row_length=None):
        """
        seq_matrix: (batch_size, max_seq_len) with paddings at the end.
        row_length: if this is not None, we will expand the row to this length.
            if this is None, we will expand to max_seq_len.
        returns: (batch_size, max_seq_len or row_length, max_seq_len) with 1s for the meaningful tokens
                    and 0s for the paddings (at the end of each row).
        """
        mask = seq_matrix != self.padding_idx
        mask = mask.type(torch.float32)
        row_length = row_length if row_length is not None else seq_matrix.shape[1]
        return mask.unsqueeze(1).expand(-1, row_length, -1)

    def _gen_autoregressive_mask(self, seq_matrix):
        """
        seq_matrix: (batch_size, max_seq_len) with paddings at the end - similar to _gen_max_seq_mask.
            this will mask out all the "future" tokens in the sequence.
        returns: (batch_size, max_seq_len, max_seq_len)
        """
        max_seq_len = seq_matrix.shape[1]
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = mask.type(torch.float32)
        return mask.unsqueeze(0).expand(seq_matrix.shape[0], -1, -1)

    def forward(self, input_ids, memory=None):
        """
        input_ids: (batch_size, max_seq_len). Each element is an integer representing the index of the token in the vocabulary.
            max_seq_len is the maximum sequence length of the input in the batch.
            all inputs are padded to this length for parallel processing.
        output_ids: (batch_size, max_seq_len). This represents the output.
        output dimension is (batch_size, output_seq_len, vocab_size).

        NOTE: padding should be added to input_ids and output_ds (and specified by padding_idx).
        TODO: who's doing the shifting?

        a simple forward pass to either predict 1 token or use in training (compare against multiple IDs).
        """
        assert input_ids.shape[1] <= self.max_seq_len, f"input sequence length {input_ids.shape[1]} is greater than max sequence length {self.max_seq_len}!"
        assert memory.shape[1] <= self.max_seq_len, f"output sequence length {memory.shape[1]} is greater than max sequence length {self.max_seq_len}!"
        input_mask = self._gen_max_seq_mask(input_ids)
        encode_decode_att_mask = self._gen_max_seq_mask(input_ids, row_length=memory.shape[1])

        memory_mask = self._gen_max_seq_mask(memory)
        memory_mask = self._gen_autoregressive_mask(memory) * memory_mask

        input_embeds = self.embedding(input_ids)
        input_embeds += self.pos_embedding(torch.arange(input_embeds.shape[1])) # (batch_size, max_seq_len, hidden_size)
        memory_embeds = self.embedding(memory) # (batch_size, max_seq_len, hidden_size)
        memory_embeds += self.pos_embedding(torch.arange(memory_embeds.shape[1])) # (batch_size, max_seq_len, hidden_size)

        encoder_output = self.encoder(input_embeds, mask=input_mask)
        # NOTE: some implementation passes in the input_mask here. I think that is wrong.
        decoder_output = self.decoder(memory_embeds, encoder_output, mask=encode_decode_att_mask, memory_mask=memory_mask)
        linear_out = self.linear(decoder_output) # (batch_size, output_seq_len, vocab_size)
        return torch.softmax(linear_out, dim=-1)

if __name__ == '__main__':
    # %%
    att = ScaledProductAttention(10, 2)
    for name, parameter in att.named_parameters():
        print(name, parameter.shape)

    # %%
    # Define the embedding matrix (2 tokens x 3 dimensions)
    E = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    E = E.unsqueeze(0)

    # Q, K, V are all the same in masked self-attention
    Q = K = V = E

    attention = ScaledProductAttention(3)
    print(attention(Q, K, V))

    # %%
    batch_size = 4
    n = 5 # input sequence length.
    m = 7 # output sequence length.
    hidden_size = 512 # hidden size of the model.
    Q = torch.randn(batch_size, n, hidden_size)
    K = torch.randn(batch_size, m, hidden_size)
    V = torch.randn(batch_size, m, hidden_size)

    attention = ScaledProductAttention(hidden_size)
    attention(Q, K, V).shape

    # %%
    # print all the parameters of attention with their variable names.
    for name, param in attention.named_parameters():
        print(name, param.shape)

    # %%
    list(attention.named_parameters())

    # %%
    hidden_size = 768
    linear_size = 2048
    batch_size = 4
    x = torch.randn(batch_size, 5, hidden_size)
    ffn = PositionWiseFFN(hidden_size, linear_size)
    ffn(x).shape

    # %%
    X = torch.randn(batch_size, 5, hidden_size)
    encoder_block = EncoderBlock(hidden_size)
    encoder_block(X).shape

    # %%
    X = torch.randn(batch_size, 5, hidden_size)
    layer_norm = LayerNorm(hidden_size)
    layer_norm(X).shape

    # %%
    X = torch.randn(batch_size, 5, hidden_size)
    encoder = Encoder(hidden_size, 6)
    encoder(X).shape

    # %%
    # input sequence length is 5 and output sequence length is 7.
    x = torch.randn(batch_size, 7, hidden_size)
    encoder_output = torch.randn(batch_size, 5, hidden_size)
    decoder_block = DecoderBlock(hidden_size)
    decoder_block(x, encoder_output).shape

    # %%
    input_seq = torch.randn(batch_size, 5, hidden_size)
    encoder = Encoder(hidden_size, 6)
    encoder_output = encoder(input_seq)
    output_seq = torch.randn(batch_size, 7, hidden_size)
    decoder = Decoder(hidden_size, 6)
    decoder(output_seq, encoder_output).shape

    # %%
    model = Transformer({'hidden_size': hidden_size, 'num_layers': 6, 'vocab_size': 1000, 'max_seq_len': 512, 'padding_idx': 0})
    input_ids = torch.randint(0, 1000, (batch_size, 5))
    input_ids[:, -2:] = 0
    memory_ids = torch.randint(0, 1000, (batch_size, 7))
    memory_ids[:, -3:] = 0
    model(input_ids, memory_ids).shape
    a = model._gen_max_seq_mask(memory_ids)
    b = model._gen_autoregressive_mask(memory_ids)
    a * b

