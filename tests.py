import spacy
import numpy as np
nlpPt = spacy.load("pt_core_news_lg")
nlpEn = spacy.load("en_core_web_lg")

your_word = "cachorro quente"


print(nlpPt(your_word).vector.shape)
for token in nlpPt(your_word):
  print(token.text)
  print(token.vector.shape)
  vector = np.array([token.vector])

exit()
print(your_word, vector)
ms = nlpPt.vocab.vectors.most_similar(vector, n=10)
words = [nlpPt.vocab.strings[w] for w in ms[0][0]]
distances = ms[2]
print(words)

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