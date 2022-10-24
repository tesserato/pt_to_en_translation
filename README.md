# pt_to_en_translation
testing transformers performance in the task of translating from Portuguese to English

# code adapted from
https://www.tensorflow.org/text/tutorials/transformer 
https://pytorch.org/tutorials/beginner/translation_transformer.html
Carlos Tarjano 21 10 2022







# '''
# Here we are defining classes (mostly helper classes), all of them inheriting from "nn.Module". Later, those classes must be instantiated to define a model
# '''
# # helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
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

# # helper Module to convert tensor of input indices into corresponding tensor of token embeddings
# class TokenEmbedding(nn.Module):
#     def __init__(self, vocab_size: int, emb_size):
#         super(TokenEmbedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_size)
#         self.emb_size = emb_size

#     def forward(self, tokens: Tensor):
#         return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# # Seq2Seq Network
# class Seq2SeqTransformer(nn.Module):
#     def __init__(self,
#                  num_encoder_layers: int,
#                  num_decoder_layers: int,
#                  emb_size: int,
#                  nhead: int,
#                  src_vocab_size: int,
#                  tgt_vocab_size: int,
#                  dim_feedforward: int = 512,
#                  dropout: float = 0.1):
#         super(Seq2SeqTransformer, self).__init__()
#         self.transformer = Transformer(d_model=emb_size,
#                                        nhead=nhead,
#                                        num_encoder_layers=num_encoder_layers,
#                                        num_decoder_layers=num_decoder_layers,
#                                        dim_feedforward=dim_feedforward,
#                                        dropout=dropout)
#         self.generator = nn.Linear(emb_size, tgt_vocab_size)
#         self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
#         self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
#         self.positional_encoding = PositionalEncoding(
#             emb_size, dropout=dropout)

#     def forward(self,
#                 src: Tensor,
#                 trg: Tensor,
#                 src_mask: Tensor,
#                 tgt_mask: Tensor,
#                 src_padding_mask: Tensor,
#                 tgt_padding_mask: Tensor,
#                 memory_key_padding_mask: Tensor):
#         src_emb = self.positional_encoding(self.src_tok_emb(src))
#         tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
#         outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
#                                 src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
#         return self.generator(outs)

#     def encode(self, src: Tensor, src_mask: Tensor):
#         return self.transformer.encoder(self.positional_encoding(
#                             self.src_tok_emb(src)), src_mask)

#     def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
#         return self.transformer.decoder(self.positional_encoding(
#                           self.tgt_tok_emb(tgt)), memory,
#                           tgt_mask)

# def train_epoch(model, optimizer):
#     model.train()
#     losses = 0
#     train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
#     train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

#     for src, tgt in train_dataloader:
#         src = src.to(DEVICE)
#         tgt = tgt.to(DEVICE)

#         tgt_input = tgt[:-1, :]

#         src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

#         logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

#         optimizer.zero_grad()

#         tgt_out = tgt[1:, :]
#         loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
#         loss.backward()

#         optimizer.step()
#         losses += loss.item()

#     return losses / len(train_dataloader)


# def evaluate(model):
#     model.eval()
#     losses = 0

#     val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
#     val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

#     for src, tgt in val_dataloader:
#         src = src.to(DEVICE)
#         tgt = tgt.to(DEVICE)

#         tgt_input = tgt[:-1, :]

#         src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

#         logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

#         tgt_out = tgt[1:, :]
#         loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
#         losses += loss.item()

#     return losses / len(val_dataloader)


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(nlpPt.vocab)
TGT_VOCAB_SIZE = len(nlpEn.vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)



for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

NUM_EPOCHS = 10

for epoch in range(1, NUM_EPOCHS+1):
    
    transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

