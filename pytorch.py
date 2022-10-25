import numpy as np
from cmath import inf
import io
from torchtext.vocab import build_vocab_from_iterator
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn.utils.rnn import pad_sequence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout, 
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # print("emb sizes", src_emb.size(), tgt_emb.size())
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        print("Encoding")
        return self.transformer.encoder(
          self.positional_encoding(self.src_tok_emb(src)), 
          src_mask
          )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        print("Decoding")
        pos_enc = self.positional_encoding(self.tgt_tok_emb(tgt))
        print(pos_enc.size(), memory.size())
        return self.transformer.decoder(
          pos_enc, 
          memory,
          tgt_mask
        )

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)#.transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX)#.transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

####################
def toTokens(string):
  return string.strip().split()

def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            yield toTokens(line)

def tensorFromRawData(file_path, vocab):
  data = []
  with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
      data.append(
        torch.tensor([BOS_IDX] + vocab(toTokens(line.rstrip("\n"))) + [EOS_IDX])
      )
  return pad_sequence(data, padding_value=PAD_IDX, batch_first=True)

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for _ in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        print("Decoded")
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = torch.tensor([BOS_IDX] + src_vocab(toTokens(src_sentence.rstrip("\n"))) + [EOS_IDX]).view(-1, 1)
    print(src.size())
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(tgt_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")



src_vocab = build_vocab_from_iterator(yield_tokens("01_train_pt.txt"), specials=special_symbols)
src_vocab.set_default_index(UNK_IDX)
tgt_vocab = build_vocab_from_iterator(yield_tokens("02_train_en.txt"), specials=special_symbols)
tgt_vocab.set_default_index(UNK_IDX)

src_train_tensor = tensorFromRawData("01_train_pt.txt", src_vocab)
tgt_train_tensor = tensorFromRawData("02_train_en.txt", tgt_vocab)
src_val_tensor = tensorFromRawData("03_val_pt.txt", src_vocab)
tgt_val_tensor = tensorFromRawData("04_val_en.txt", tgt_vocab)

print(src_train_tensor.size(), tgt_train_tensor.size(), src_val_tensor.size(), tgt_val_tensor.size())


torch.manual_seed(0)
EPOCHS = 1000
SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 8 #128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

train = torch.utils.data.TensorDataset(src_train_tensor, tgt_train_tensor)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val = torch.utils.data.TensorDataset(src_val_tensor, tgt_val_tensor)
val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)


transformer = Seq2SeqTransformer(
  NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM
                                 )

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

print("TEST:", translate(transformer, "mas e se estes fatores fossem ativos ?"))

#########################

bestEvalLoss = inf
for epoch in range(EPOCHS):
  print(f"#### Epoch {epoch} ####")
  transformer.train()
  for iter, (src, tgt) in enumerate(train_loader):
      src = src.to(DEVICE)
      tgt = tgt.to(DEVICE)
      tgt_input = tgt[:, :-1]

      # print("ipt sizes", src.size(), tgt_input.size())

      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

      logits = transformer(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

      optimizer.zero_grad()

      tgt_out = tgt[:, 1:]
      loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
      loss.backward()
      optimizer.step()
      print(loss.item())
      if (iter + 1) % 5 == 0:
        print("Evaluating:")
        transformer.eval()
        totalEvalLoss = []
        for iterVal, (src, tgt) in enumerate(val_loader):
          src = src.to(DEVICE)
          tgt = tgt.to(DEVICE)
          tgt_input = tgt[:, :-1]

          # print("ipt sizes", src.size(), tgt_input.size())

          src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

          logits = transformer(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

          optimizer.zero_grad()

          tgt_out = tgt[:, 1:]
          loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
          loss.backward()
          optimizer.step()
          totalEvalLoss.append(loss.item())
          if iterVal > 2:
            break
        evalLoss = np.average(totalEvalLoss)
        print(f"    Eval average loss = {evalLoss}")
        if evalLoss < bestEvalLoss:
          print(f"    Improved from {bestEvalLoss} to {evalLoss} average validation loss")
          bestEvalLoss = evalLoss