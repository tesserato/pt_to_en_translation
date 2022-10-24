import tensorflow_datasets as tfds
import spacy
# import numpy as np
nlpEn = spacy.load("en_core_web_sm")
nlpPt = spacy.load("pt_core_news_lg")
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running in {DEVICE} mode")

print(f"len pt vocab = {len(nlpPt.vocab)}, len en vocab = {len(nlpEn.vocab)}")

'''This will download the dataset on first run'''
examples, metadata = tfds.load(
    'ted_hrlr_translate/pt_to_en',
    with_info=True,
    as_supervised=True
)
# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
# class PositionalEncoding(nn.Module):
#     def __init__(self,
#                  emb_size: int,
#                  dropout: float,
#                  maxlen: int = 5000):
#         super(PositionalEncoding, self).__init__()
#         den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)

#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', pos_embedding)

#     def forward(self, token_embedding: Tensor):
#         return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
# class TokenEmbedding(nn.Module):
#     def __init__(self, vocab_size: int, emb_size):
#         super(TokenEmbedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_size)
#         self.emb_size = emb_size

#     def forward(self, tokens: Tensor):
#         return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                #  src_vocab_size: int,
                #  tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        # self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor = None,
                tgt_mask: Tensor = None,
                src_padding_mask: Tensor = None,
                tgt_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None):
        # src_emb = self.positional_encoding(self.src_tok_emb(src))
        # tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src, tgt, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return outs #self.generator(outs)

    # def encode(self, src: Tensor, src_mask: Tensor):
    #     return self.transformer.encoder(self.positional_encoding(
    #                         self.src_tok_emb(src)), src_mask)

    # def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
    #     return self.transformer.decoder(self.positional_encoding(
    #                       self.tgt_tok_emb(tgt)), memory,
    #                       tgt_mask)

# def generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask


# def create_mask(src, tgt):
#     src_seq_len = src.shape[0]
#     tgt_seq_len = tgt.shape[0]

#     tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
#     src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

#     src_padding_mask = (src == PAD_IDX).transpose(0, 1)
#     tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
#     return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


d_model = 64
transformer = Seq2SeqTransformer(
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 emb_size=d_model,
                 nhead=8,
                #  src_vocab_size=len(nlpPt.vocab),
                #  tgt_vocab_size=len(nlpEn.vocab),
)


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


train_examples, val_examples = examples['train'], examples['validation']
# pt_examples, en_examples are arrays of 3 bytestrings representing a phrase in portuguese and its translation in english
for batch, (pt_examples, en_examples) in enumerate(train_examples.batch(1).take(200)):
  # print(f"########################## Batch {batch + 1} ##########################")
  for phrase, (pt, en) in enumerate(zip(pt_examples.numpy(), en_examples.numpy())):
    src = nlpPt(pt.decode('utf-8')).vector
    tgt = nlpEn(en.decode('utf-8')).vector
    m = min(len(src), len(tgt))
    src = torch.tensor(src[:d_model]).unsqueeze(0).unsqueeze(0)
    tgt = torch.tensor(tgt[:d_model]).unsqueeze(0).unsqueeze(0)
    # print(src.shape, tgt.shape)
    out = transformer.forward(src, tgt)
    loss = loss_fn(out, tgt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.float())


# print(f"Phrase {phrase} pt: {pt_utf8}")
# print(nlpPt(pt_utf8).vector)
# print(",".join([f"{token.text}:{token.idx}" for token in nlpPt(pt_utf8)]))
# print(f"Phrase {phrase} en: {en_utf8}")
# print(",".join([f"{token.text}:{token.idx}" for token in nlpEn(en_utf8)]))
# print()

# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# for epoch in range(10):
#   out = transformer.forward(src, tgt)
#   loss = loss_fn(out, tgt)
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#   print(loss)
# exit()




# your_word = "cachorro"

# for token in nlpPt(your_word):
#   vector = np.array([token.vector])
#   break
# print(your_word, vector)
# ms = nlpPt.vocab.vectors.most_similar(vector, n=10)
# words = [nlpPt.vocab.strings[w] for w in ms[0][0]]
# distances = ms[2]
# print(words)
# exit()



